"""N-episode experiment runner for Go2 teleoperation.

Wraps the real-time control loop from run_teleop_new.py / run_quest3.py in a
multi-episode loop.  The robot is controlled by the HUMAN OPERATOR via the
glove or Quest 3 controller — no scripted velocity commands.

Between episodes the MuJoCo simulation is fully reset to the 'home' keyframe
(mj_resetDataKeyframe) so every run starts from the same standing pose.

Usage
-----
    # Glove controller (serial):
    python teleop/run_experiment.py --controller glove --runs 5
    python teleop/run_experiment.py --controller glove --port /dev/ttyACM1

    # Quest 3 controller (OpenVR / ALVR):
    python teleop/run_experiment.py --controller quest3 --runs 5

    # Common options:
    python teleop/run_experiment.py --controller glove --runs 10 --max-time 60
    python teleop/run_experiment.py --controller quest3 --scene scene_square_walled

Goal detection
--------------
    Scene                    Goal position
    scene_zigzag_walled      wp_5   = (10, 0)  — point-to-point
    scene_zigzag_short       wp_3   = ( 6, 0)  — point-to-point
    scene_small_zigzag_walled  wp_3 = ( 6, 0)  — point-to-point (small)
    scene_square_walled      spawn  = ( 0, 0)  — full loop
    scene_circle_walled      spawn  = ( 0, 0)  — full loop
    scene_small_square_walled  spawn = ( 0, 0)  — full loop (4×4 m)
    scene_small_circle_walled  spawn = ( 0, 0)  — full loop (r=2 m)
    Override with --goal-x / --goal-y.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import mujoco as mj
import mujoco.viewer
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model   import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait            import Gait

from teleop.feature_trigger import RobotFeatureManager


# ── Timing constants (identical to run_teleop_new.py / run_quest3.py) ────────

GAIT_HZ    = 3
GAIT_DUTY  = 0.6
GAIT_T     = 1.0 / GAIT_HZ

SIM_HZ     = 1000
SIM_DT     = 1.0 / SIM_HZ

CTRL_HZ    = 200
CTRL_DT    = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ

MPC_DT        = GAIT_T / 16
STEPS_PER_MPC = max(1, int(CTRL_HZ // (1.0 / MPC_DT)))

RENDER_HZ = 60.0
RENDER_DT = 1.0 / RENDER_HZ


_COMBO_LABEL = {
    0x00: "---",
    0x01: "M--",
    0x02: "-R-",
    0x03: "MR-",
    0x04: "--P",
    0x06: "-RP",
    0x07: "MRP",
}

_SAFETY = 0.9
TAU_LIM = _SAFETY * np.array([
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
])
LEG_SLICE = {"FL": slice(0, 3), "FR": slice(3, 6),
             "RL": slice(6, 9), "RR": slice(9, 12)}

SCENE_GOALS = {
    "scene_zigzag_walled":        (10.0, 0.0),  # wp_5 at end of zigzag
    "scene_zigzag_short":         ( 6.0, 0.0),  # wp_3 at end of short zigzag
    "scene_small_zigzag_walled":  ( 6.0, 0.0),  # wp_3 at end of small zigzag
    "scene_square_walled":        ( 0.0, 0.0),  # full loop — returns to spawn
    "scene_circle_walled":        ( 0.0, 0.0),  # full loop — returns to spawn
    "scene_small_square_walled":  ( 0.0, 0.0),  # full loop — returns to spawn
    "scene_small_circle_walled":  ( 0.0, 0.0),  # full loop — returns to spawn
}

# Minimum distance (m) the robot must have been from the goal before
# goal detection is armed.  Prevents false triggers on loop courses
# where the goal position equals the spawn position.
SCENE_GOAL_FAR_DIST = {
    "scene_square_walled":        3.0,
    "scene_circle_walled":        3.0,
    "scene_small_square_walled":  2.0,  # 4 m side — 2 m is mid-first-side
    "scene_small_circle_walled":  2.0,  # r=2 m — 2 m is ~60° into lap
}


# ── Episode result ────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode:       int
    controller:    str
    goal_reached:  bool
    wall_touched:  bool
    sim_time:      float
    wall_contacts: int
    final_dist:    float

    @property
    def clean_success(self) -> bool:
        return self.goal_reached and not self.wall_touched

    def summary(self) -> str:
        status = ("CLEAN  " if self.clean_success
                  else "GOAL+W " if self.goal_reached
                  else "TIMEOUT")
        return (f"ep {self.episode:3d}  {status}  "
                f"sim={self.sim_time:6.2f}s  "
                f"bumps={self.wall_contacts:3d}  "
                f"dist={self.final_dist:.3f}m")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_wall_geom_ids(model: mj.MjModel) -> frozenset:
    return frozenset(
        model.geom(i).id
        for i in range(model.ngeom)
        if model.geom(i).name.startswith("wall_")
    )


def _base_xy(mujoco_go2: MuJoCo_GO2_Model) -> np.ndarray:
    return mujoco_go2.data.xpos[mujoco_go2.model.body("base_link").id, :2].copy()



def _reset_episode(
    mujoco_go2: MuJoCo_GO2_Model,
    go2:        PinGo2Model,
    key_id:     int,
    z_pos:      float,
) -> tuple:
    """Reset MuJoCo + Pinocchio, return fresh (gait, traj, mpc, leg_ctrl, U_opt)."""
    # ── MuJoCo reset ──────────────────────────────────────────────────────────
    mj.mj_resetDataKeyframe(mujoco_go2.model, mujoco_go2.data, key_id)
    mj.mj_forward(mujoco_go2.model, mujoco_go2.data)

    # ── Sync Pinocchio BEFORE building ComTraj / CentroidalMPC ───────────────
    # update_pin_with_mujoco must run first so the MPC objects are initialised
    # from the correct reset state, not from the previous episode's final pose.
    mujoco_go2.update_pin_with_mujoco(go2)

    gait     = Gait(GAIT_HZ, GAIT_DUTY)
    traj     = ComTraj(go2)
    traj.generate_traj(go2, gait, 0.0, 0.0, 0.0, z_pos, 0.0, time_step=MPC_DT)
    mpc      = CentroidalMPC(go2, traj)
    leg_ctrl = LegController()
    U_opt    = np.zeros((12, traj.N), dtype=float)

    return gait, traj, mpc, leg_ctrl, U_opt


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-episode teleoperation experiment (glove or Quest 3)"
    )
    parser.add_argument("--controller", default="glove",
                        choices=["glove", "quest3"],
                        help="Input device (default: glove)")
    parser.add_argument("--scene",      default="scene_zigzag_walled",
                        help="MJCF scene name without .xml")
    parser.add_argument("--runs",       type=int,   default=5,
                        help="Number of episodes (default: 5)")
    parser.add_argument("--max-time",   type=float, default=120.0,
                        help="Episode timeout in sim-seconds (default: 120)")
    parser.add_argument("--z-pos",      type=float, default=0.27,
                        help="Standing COM height in m (default: 0.27)")
    parser.add_argument("--goal-x",     type=float, default=None)
    parser.add_argument("--goal-y",     type=float, default=None)
    parser.add_argument("--goal-radius",type=float, default=0.5,
                        help="Goal acceptance radius in m (default: 0.5)")
    parser.add_argument("--goal-far-dist", type=float, default=None,
                        help="Robot must reach this distance from goal before "
                             "goal detection arms (auto-set per scene if omitted)")
    # Glove-only
    parser.add_argument("--port",   default="/dev/ttyACM0",
                        help="Glove serial port (default: /dev/ttyACM0)")
    parser.add_argument("--mode",   default="holonomic",
                        choices=["holonomic", "differential"])
    parser.add_argument("--maxv",   type=float, default=0.8)
    parser.add_argument("--force",  action="store_true",
                        help="Force finger combo to MRP (0x07=locomotion+euler tilt) "
                             "regardless of actual finger bending")
    parser.add_argument("--maxw",   type=float, default=1.5)
    # Quest3-only (no extra args beyond --controller)
    args = parser.parse_args()

    # ── Scene ────────────────────────────────────────────────────────────────
    scene_xml = os.path.join(_ROOT, "models", "MJCF", "go2", args.scene + ".xml")
    if not os.path.exists(scene_xml):
        print(f"ERROR: scene not found: {scene_xml}")
        sys.exit(1)

    default_goal = SCENE_GOALS.get(args.scene, (10.0, 0.0))
    goal_xy = np.array([
        args.goal_x if args.goal_x is not None else default_goal[0],
        args.goal_y if args.goal_y is not None else default_goal[1],
    ])
    # Arm distance: robot must reach this far from goal before detection fires.
    # Prevents loop courses (circle/square) from triggering at spawn time.
    goal_far_dist = (args.goal_far_dist if args.goal_far_dist is not None
                     else SCENE_GOAL_FAR_DIST.get(args.scene, 0.0))

    # ── Robot model ───────────────────────────────────────────────────────────
    print(f"Loading scene : {scene_xml}")
    go2        = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model()
    mujoco_go2.model = mj.MjModel.from_xml_path(scene_xml)
    mujoco_go2.data  = mj.MjData(mujoco_go2.model)
    mujoco_go2.model.opt.timestep = SIM_DT

    key_id = mj.mj_name2id(mujoco_go2.model, mj.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        print("ERROR: 'home' keyframe not found in model")
        sys.exit(1)

    wall_ids = _collect_wall_geom_ids(mujoco_go2.model)

    # ── Controller ────────────────────────────────────────────────────────────
    if args.controller == "glove":
        from teleop.glove_controller import GloveController, GloveConfig
        cfg        = GloveConfig(port=args.port, control_mode=args.mode,
                                 max_lin_vel=args.maxv, max_ang_vel=args.maxw)
        controller = GloveController(cfg)
        feat       = RobotFeatureManager(ctrl_dt=CTRL_DT)
    else:
        from teleop.quest3_controller import QuestController, QuestConfig
        cfg        = QuestConfig(max_lin_vel=args.maxv if hasattr(args, 'maxv') else 0.8,
                                 max_ang_vel=args.maxw if hasattr(args, 'maxw') else 1.5)
        controller = QuestController(cfg)
        feat       = RobotFeatureManager(ctrl_dt=CTRL_DT)

    connected = controller.start()
    if not connected:
        print(f"WARNING: {args.controller} not connected — robot will stand still")

    # ── Banner ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print(f"Go2 Experiment  |  controller={args.controller}  runs={args.runs}")
    print(f"  Scene   : {args.scene}")
    print(f"  Goal    : {goal_xy}  radius={args.goal_radius} m  "
          f"arm_dist={goal_far_dist} m")
    print(f"  Timeout : {args.max_time} s per episode")
    print(f"  Walls   : {len(wall_ids)} geoms tracked for contact")
    if args.controller == "glove":
        print(f"  Port    : {args.port}  Mode: {args.mode}  "
              f"MaxV: {args.maxv}  MaxW: {args.maxw}")
    print("  Close viewer window or Ctrl-C to stop early.")
    print("─" * 70)

    # ── Camera follow toggle ──────────────────────────────────────────────────
    # 0 = FREE  |  1 = 3rd-person behind  |  2 = 1st-person (head cam)
    _cam_mode = [1]
    _CAM_LABELS = {0: "FREE (manual)", 1: "3rd-person (behind)", 2: "1st-person (head cam)"}
    GLFW_KEY_T  = 84

    def _apply_cam_mode(mode: int) -> None:
        if mode == 1:
            viewer.cam.type        = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco_go2.model.body("base_link").id
            viewer.cam.distance    = 2.5
            viewer.cam.elevation   = -20.0
        elif mode == 2:
            viewer.cam.type       = mj.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco_go2.model.camera("head_cam").id
        else:
            viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE

    def _key_callback(keycode: int) -> None:
        if keycode != GLFW_KEY_T:
            return
        _cam_mode[0] = (_cam_mode[0] + 1) % 3
        _apply_cam_mode(_cam_mode[0])
        print(f"\nCamera: {_CAM_LABELS[_cam_mode[0]]}")

    # ── Results accumulator ───────────────────────────────────────────────────
    all_results: List[EpisodeResult] = []

    # ── Viewer + episode loop ─────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        mujoco_go2.model, mujoco_go2.data,
        show_left_ui=False, show_right_ui=False,
        key_callback=_key_callback,
    ) as viewer:
        _apply_cam_mode(_cam_mode[0])

        try:
            for ep in range(args.runs):
                if not viewer.is_running():
                    break

                # ── Reset ─────────────────────────────────────────────────
                print(f"\n{'─'*60}")
                print(f"  Episode {ep + 1} / {args.runs}  —  resetting ...")
                gait, traj, mpc, leg_ctrl, U_opt = _reset_episode(
                    mujoco_go2, go2, key_id, args.z_pos
                )
                # Reset feature manager state (posture, triggers, etc.)
                feat = RobotFeatureManager(ctrl_dt=CTRL_DT)

                # Brief countdown so the operator knows a new run started
                for cnt in (3, 2, 1):
                    if not viewer.is_running():
                        break
                    print(f"  Starting in {cnt}...", end="\r", flush=True)
                    viewer.sync()
                    time.sleep(1.0)
                print(f"  GO!  (timeout {args.max_time} s)            ")

                # ── Per-episode state ──────────────────────────────────────
                k = ctrl_i = 0
                tau_hold    = np.zeros(12, dtype=float)
                next_render = 0.0
                wall_start  = time.perf_counter()
                # wall_contacts counts distinct contact EVENTS (rising edges),
                # not raw physics contact detections per step.
                wall_contacts    = 0
                wall_touched     = False
                _prev_wall_touch = False   # contact state from previous step
                goal_reached  = False
                # goal_armed: False until robot has been >= goal_far_dist from
                # goal. Prevents loop courses from triggering at spawn time.
                goal_armed    = (goal_far_dist == 0.0)

                # ── Real-time simulation loop ──────────────────────────────
                while viewer.is_running():
                    sim_time = float(mujoco_go2.data.time)

                    # Real-time pacing (same as run_teleop.py)
                    ahead = sim_time - (time.perf_counter() - wall_start)
                    if ahead > 0.0:
                        time.sleep(ahead)

                    # ── Episode termination ────────────────────────────────
                    if sim_time >= args.max_time:
                        break
                    dist_to_goal = float(np.linalg.norm(_base_xy(mujoco_go2) - goal_xy))
                    if not goal_armed and dist_to_goal >= goal_far_dist:
                        goal_armed = True
                    if goal_armed and dist_to_goal < args.goal_radius:
                        goal_reached = True
                        break

                    # ── Control tick ───────────────────────────────────────
                    if k % CTRL_DECIM == 0:
                        # Read controller
                        combo      = controller.get_finger_combo()
                        if args.force:
                            combo = 0x07  # MRP : force locomotion + euler tilt
                        raw_vx, raw_vy, raw_wz = controller.get_velocity_command()
                        ctrl_state = controller.get_state()
                        hovering = controller.is_hovering()

                        # Feature manager → final commands
                        cmd = feat.update(combo, raw_vx, raw_vy, raw_wz,
                                          hovering, sim_time)
                        x_vel = cmd["x_vel"]
                        y_vel = cmd["y_vel"]
                        ang_z = cmd["ang_z"]
                        z_pos = cmd["z_pos"]
                        euler_shift = cmd["euler_shift"]

                        # Body-frame velocity mapping (mirrors run_teleop_new.py / run_quest3.py)
                        x_vel_b = x_vel
                        y_vel_b = y_vel
                        _speed_b = np.hypot(x_vel_b, y_vel_b)

                        if args.controller == "glove":
                            if euler_shift:
                                # MRP: side component → ang_z, no strafe
                                ang_z = float(np.clip(
                                    cfg.max_ang_vel * (y_vel_b / _speed_b if _speed_b > 0.01 else 0.0),
                                    -cfg.max_ang_vel, cfg.max_ang_vel,
                                ))
                                y_vel_b = 0.0
                            else:
                                # MR: holonomic movement, no yaw
                                ang_z = 0.0
                        else:  # quest3
                            if euler_shift:
                                ang_z = 0.0  # MRP = fixed-orientation strafe

                        mujoco_go2.update_pin_with_mujoco(go2)

                        # MPC solve
                        if ctrl_i % STEPS_PER_MPC == 0:
                            traj.generate_traj(
                                go2, gait, sim_time,
                                x_vel_b, y_vel_b, z_pos, ang_z,
                                time_step=MPC_DT,
                            )

                            sol   = mpc.solve_QP(go2, traj, False)
                            N     = traj.N
                            w_opt = sol["x"].full().flatten()
                            U_opt = w_opt[12 * N:].reshape((12, N), order="F")

                            print(
                                f"\r  t={sim_time:5.1f}s  "
                                f"{ctrl_state:4s}  "
                                f"[{_COMBO_LABEL.get(combo, f'{combo:03b}')}]  "
                                f"vx:{x_vel:+.2f} vy:{y_vel:+.2f} wz:{ang_z:+.2f}  "
                                f"bumps={wall_contacts}  dist={dist_to_goal:.2f}m   ",
                                end="", flush=True,
                            )

                        mpc_force = U_opt[:, 0]
                        tau_raw   = np.zeros(12)
                        for leg in ("FL", "FR", "RL", "RR"):
                            out = leg_ctrl.compute_leg_torque(
                                leg, go2, gait, mpc_force[LEG_SLICE[leg]], sim_time,
                            )
                            tau_raw[LEG_SLICE[leg]] = out.tau
                        tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
                        ctrl_i += 1

                    # ── Physics step ───────────────────────────────────────
                    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
                    mujoco_go2.set_joint_torque(tau_hold)
                    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)
                    k += 1

                    # ── Wall contact check (rising-edge events only) ───────
                    _cur_wall_touch = any(
                        mujoco_go2.data.contact[i].geom1 in wall_ids or
                        mujoco_go2.data.contact[i].geom2 in wall_ids
                        for i in range(mujoco_go2.data.ncon)
                    )
                    if _cur_wall_touch:
                        wall_touched = True
                        if not _prev_wall_touch:   # rising edge → new event
                            wall_contacts += 1
                    _prev_wall_touch = _cur_wall_touch

                    # ── Render ─────────────────────────────────────────────
                    if sim_time >= next_render:
                        if _cam_mode[0] == 1:
                            _qw, _qx, _qy, _qz = mujoco_go2.data.qpos[3:7]
                            _yaw = np.arctan2(
                                2.0 * (_qw * _qz + _qx * _qy),
                                1.0 - 2.0 * (_qy ** 2 + _qz ** 2),
                            )
                            viewer.cam.azimuth = np.degrees(_yaw)
                        viewer.sync()
                        next_render += RENDER_DT

                # ── Episode done ───────────────────────────────────────────
                result = EpisodeResult(
                    episode      = ep,
                    controller   = args.controller,
                    goal_reached = goal_reached,
                    wall_touched = wall_touched,
                    sim_time     = float(mujoco_go2.data.time),
                    wall_contacts= wall_contacts,
                    final_dist   = float(np.linalg.norm(
                                       _base_xy(mujoco_go2) - goal_xy)),
                )
                all_results.append(result)
                print(f"\n  {result.summary()}")

        except KeyboardInterrupt:
            print("\n\nCtrl-C — stopping early.")
        finally:
            controller.stop()

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(all_results, args.controller)


def _print_summary(results: List[EpisodeResult], ctrl: str) -> None:
    n = len(results)
    if n == 0:
        print("No episodes completed.")
        return
    clean   = sum(1 for r in results if r.clean_success)
    reached = sum(1 for r in results if r.goal_reached)
    touched = sum(1 for r in results if r.wall_touched)
    print()
    print("=" * 56)
    print(f"  Experiment summary — {ctrl.upper()}  ({n} episodes)")
    print("=" * 56)
    print(f"  Clean success  (goal + no wall) : {clean:3d}/{n}  ({clean/n:.1%})")
    print(f"  Goal reached                    : {reached:3d}/{n}  ({reached/n:.1%})")
    print(f"  Wall contacted (any)            : {touched:3d}/{n}  ({touched/n:.1%})")
    print(f"  Avg sim time                    : {np.mean([r.sim_time      for r in results]):.2f} s")
    print(f"  Avg wall events / episode       : {np.mean([r.wall_contacts for r in results]):.1f}")
    print(f"  Avg final dist to goal          : {np.mean([r.final_dist    for r in results]):.3f} m")
    print("=" * 56)
    print(f"  {'Ep':>3}  {'Goal':5}  {'Wall':5}  {'WEvt':6}  {'SimT':7}  {'Dist':8}")
    print(f"  {'-'*44}")
    for r in results:
        print(f"  {r.episode+1:3d}  "
              f"{'Y' if r.goal_reached else 'n':5}  "
              f"{'Y' if r.wall_touched  else 'n':5}  "
              f"{r.wall_contacts:6d}  "
              f"{r.sim_time:7.2f}  "
              f"{r.final_dist:8.3f}")
    print()


if __name__ == "__main__":
    main()
