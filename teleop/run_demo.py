"""Poster-session demo runner — infinite-episode glove teleoperation.

Runs the Go2 MPC loop continuously, resetting the simulation after each
episode (goal reached or operator skips).  A matplotlib dashboard shows
the glove hand state, finger bend sensors, magnetometer distance gauge,
and live velocity.

Usage
-----
    python teleop/run_demo.py

    # Override scene:
    python teleop/run_demo.py --scene scene_small_square_walled

    # Adjust velocity limits:
    python teleop/run_demo.py --maxv 0.6 --maxw 1.2

    # Force MRP combo (loco + lean) regardless of finger bending:
    python teleop/run_demo.py --force

    # Per-episode time limit (seconds), default = unlimited:
    python teleop/run_demo.py --max-time 90

Keyboard (viewer window)
------------------------
    T — cycle camera mode: 3rd-person → 1st-person → free
    N — skip to next episode immediately
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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
from teleop.demo_hud        import DemoState, DemoHUD


# ── Timing constants (identical to run_experiment.py) ────────────────────────

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

_SAFETY = 0.9
TAU_LIM = _SAFETY * np.array([
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
])
LEG_SLICE = {"FL": slice(0, 3), "FR": slice(3, 6),
             "RL": slice(6, 9), "RR": slice(9, 12)}

_COMBO_LABEL = {
    0x00: "---", 0x01: "M--", 0x02: "-R-", 0x03: "MR-",
    0x04: "--P", 0x06: "-RP", 0x07: "MRP",
}

# Default goals per scene (same as run_experiment.py)
SCENE_GOALS = {
    "scene_zigzag_walled":        (10.0, 0.0),
    "scene_zigzag_short":         ( 6.0, 0.0),
    "scene_small_zigzag_walled":  ( 6.0, 0.0),
    "scene_square_walled":        ( 0.0, 0.0),
    "scene_circle_walled":        ( 0.0, 0.0),
    "scene_small_square_walled":  ( 0.0, 0.0),
    "scene_small_circle_walled":  ( 0.0, 0.0),
}
SCENE_GOAL_FAR_DIST = {
    "scene_square_walled":        3.0,
    "scene_circle_walled":        3.0,
    "scene_small_square_walled":  2.0,
    "scene_small_circle_walled":  2.0,
}


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
    mj.mj_resetDataKeyframe(mujoco_go2.model, mujoco_go2.data, key_id)
    mj.mj_forward(mujoco_go2.model, mujoco_go2.data)
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
        description="Infinite-episode poster demo — glove teleoperation"
    )
    parser.add_argument("--scene",       default="scene_small_zigzag_walled",
                        help="MJCF scene name without .xml "
                             "(default: scene_small_zigzag_walled)")
    parser.add_argument("--max-time",    type=float, default=None,
                        help="Episode timeout in sim-seconds (default: unlimited)")
    parser.add_argument("--z-pos",       type=float, default=0.27)
    parser.add_argument("--goal-x",      type=float, default=None)
    parser.add_argument("--goal-y",      type=float, default=None)
    parser.add_argument("--goal-radius", type=float, default=0.5)
    parser.add_argument("--goal-far-dist", type=float, default=None)
    # Glove
    parser.add_argument("--port",   default="/dev/ttyACM1")
    parser.add_argument("--mode",   default="holonomic",
                        choices=["holonomic", "differential"])
    parser.add_argument("--maxv",   type=float, default=0.8)
    parser.add_argument("--maxw",   type=float, default=1.5)
    parser.add_argument("--force",  action="store_true",
                        help="Force finger combo to MRP (loco+lean) always")
    args = parser.parse_args()

    # ── Scene ─────────────────────────────────────────────────────────────
    scene_xml = os.path.join(_ROOT, "models", "MJCF", "go2", args.scene + ".xml")
    if not os.path.exists(scene_xml):
        print(f"ERROR: scene not found: {scene_xml}")
        sys.exit(1)

    default_goal = SCENE_GOALS.get(args.scene, (6.0, 0.0))
    goal_xy = np.array([
        args.goal_x if args.goal_x is not None else default_goal[0],
        args.goal_y if args.goal_y is not None else default_goal[1],
    ])
    goal_far_dist = (args.goal_far_dist if args.goal_far_dist is not None
                     else SCENE_GOAL_FAR_DIST.get(args.scene, 0.0))

    # ── Robot model ───────────────────────────────────────────────────────
    print(f"Loading scene: {scene_xml}")
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

    # ── Glove controller ──────────────────────────────────────────────────
    from teleop.glove_controller import GloveController, GloveConfig
    cfg        = GloveConfig(port=args.port, control_mode=args.mode,
                             max_lin_vel=args.maxv, max_ang_vel=args.maxw)
    controller = GloveController(cfg)
    connected  = controller.start()
    if not connected:
        print("WARNING: glove not connected — robot will stand still")

    # ── Demo HUD ──────────────────────────────────────────────────────────
    demo_state = DemoState()
    hud        = DemoHUD(demo_state)
    hud.start()
    print("HUD started — see 'Go2 Glove Dashboard' window.")

    # ── Banner ────────────────────────────────────────────────────────────
    timeout_str = f"{args.max_time} s" if args.max_time is not None else "unlimited"
    print("─" * 65)
    print("Go2 Poster Demo  |  glove  |  infinite episodes")
    print(f"  Scene   : {args.scene}")
    print(f"  Goal    : {goal_xy}  radius={args.goal_radius} m")
    print(f"  Timeout : {timeout_str} per episode")
    print(f"  Port    : {args.port}  Mode: {args.mode}  "
          f"MaxV: {args.maxv}  MaxW: {args.maxw}")
    print("  Viewer keys:  T = cycle camera   N = skip episode")
    print("  Close viewer window or Ctrl-C to stop.")
    print("─" * 65)

    # ── Camera + skip-episode shared state ────────────────────────────────
    _cam_mode = [1]   # 0=free, 1=3rd-person, 2=1st-person
    _skip_ep  = [False]
    GLFW_KEY_T, GLFW_KEY_N = 84, 78

    def _apply_cam(mode: int) -> None:
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
        if keycode == GLFW_KEY_T:
            _cam_mode[0] = (_cam_mode[0] + 1) % 3
            _apply_cam(_cam_mode[0])
        elif keycode == GLFW_KEY_N:
            _skip_ep[0] = True

    # ── Episode loop ──────────────────────────────────────────────────────
    ep = 0
    with mujoco.viewer.launch_passive(
        mujoco_go2.model, mujoco_go2.data,
        show_left_ui=False, show_right_ui=False,
        key_callback=_key_callback,
    ) as viewer:
        _apply_cam(_cam_mode[0])

        try:
            while viewer.is_running():
                ep += 1
                demo_state.update(episode=ep, wall_bumps=0, dist_to_goal=0.0)

                # ── Reset ─────────────────────────────────────────────────
                print(f"\n{'─'*55}")
                print(f"  Episode {ep}  —  resetting simulation …")
                gait, traj, mpc, leg_ctrl, U_opt = _reset_episode(
                    mujoco_go2, go2, key_id, args.z_pos
                )
                feat = RobotFeatureManager(ctrl_dt=CTRL_DT)
                _skip_ep[0] = False

                for cnt in (3, 2, 1):
                    if not viewer.is_running():
                        break
                    print(f"  Starting in {cnt}…", end="\r", flush=True)
                    viewer.sync()
                    time.sleep(1.0)

                timeout_label = (f"{args.max_time:.0f} s"
                                 if args.max_time is not None else "∞")
                print(f"  GO!  (timeout {timeout_label})              ")

                # ── Per-episode state ──────────────────────────────────────
                k = ctrl_i = 0
                tau_hold     = np.zeros(12, dtype=float)
                next_render  = 0.0
                wall_start   = time.perf_counter()
                wall_contacts    = 0
                wall_touched     = False
                _prev_wall_touch = False
                goal_reached     = False
                goal_armed       = (goal_far_dist == 0.0)

                # ── Real-time simulation loop ──────────────────────────────
                while viewer.is_running():
                    sim_time = float(mujoco_go2.data.time)

                    # Real-time pacing
                    ahead = sim_time - (time.perf_counter() - wall_start)
                    if ahead > 0.0:
                        time.sleep(ahead)

                    # ── Episode termination ────────────────────────────────
                    if _skip_ep[0]:
                        print("\n  [N] skipped by operator")
                        break
                    if args.max_time is not None and sim_time >= args.max_time:
                        print(f"\n  Timeout ({args.max_time} s)")
                        break
                    dist_to_goal = float(
                        np.linalg.norm(_base_xy(mujoco_go2) - goal_xy)
                    )
                    if not goal_armed and dist_to_goal >= goal_far_dist:
                        goal_armed = True
                    if goal_armed and dist_to_goal < args.goal_radius:
                        goal_reached = True
                        break

                    # ── Control tick ───────────────────────────────────────
                    if k % CTRL_DECIM == 0:
                        combo      = controller.get_finger_combo()
                        if args.force:
                            combo  = 0x07
                        raw_vx, raw_vy, raw_wz = controller.get_velocity_command()
                        ctrl_state = controller.get_state()
                        hovering   = controller.is_hovering()

                        # Raw sensor data for HUD (glove-specific)
                        raw         = controller.get_raw()
                        finger_bent = raw['finger_bent']   # [Middle, Ring, Pinky]
                        mag_value   = float(np.sqrt(
                            raw['x_corr']**2 + raw['y_corr']**2 + raw['z_corr']**2
                        ))

                        # Feature manager → final commands
                        cmd     = feat.update(combo, raw_vx, raw_vy, raw_wz,
                                              hovering, sim_time)
                        x_vel   = cmd["x_vel"]
                        y_vel   = cmd["y_vel"]
                        ang_z   = cmd["ang_z"]
                        z_pos   = cmd["z_pos"]
                        euler_shift = cmd["euler_shift"]

                        # Body-frame velocity (mirrors run_experiment.py)
                        x_vel_b = x_vel
                        y_vel_b = y_vel
                        _speed_b = np.hypot(x_vel_b, y_vel_b)

                        if euler_shift:
                            # MRP: side component → yaw, no strafe
                            ang_z = float(np.clip(
                                cfg.max_ang_vel * (
                                    y_vel_b / _speed_b if _speed_b > 0.01 else 0.0
                                ),
                                -cfg.max_ang_vel, cfg.max_ang_vel,
                            ))
                            y_vel_b = 0.0
                        else:
                            ang_z = 0.0   # MR: holonomic, no yaw

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
                                f"{ctrl_state:8s}  "
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
                        ctrl_i  += 1

                        # Update HUD state
                        demo_state.update(
                            sim_time     = sim_time,
                            glove_state  = ctrl_state,
                            combo        = combo,
                            finger_bent  = finger_bent,
                            mag_value    = mag_value,
                            vx           = x_vel,
                            vy           = y_vel,
                            wz           = ang_z,
                            status       = _COMBO_LABEL.get(combo, "---"),
                            wall_bumps   = wall_contacts,
                            dist_to_goal = dist_to_goal,
                        )

                    # ── Physics step ───────────────────────────────────────
                    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
                    mujoco_go2.set_joint_torque(tau_hold)
                    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)
                    k += 1

                    # ── Wall contact (rising-edge events) ─────────────────
                    _cur = any(
                        mujoco_go2.data.contact[i].geom1 in wall_ids or
                        mujoco_go2.data.contact[i].geom2 in wall_ids
                        for i in range(mujoco_go2.data.ncon)
                    )
                    if _cur:
                        wall_touched = True
                        if not _prev_wall_touch:
                            wall_contacts += 1
                    _prev_wall_touch = _cur

                    # ── Render ─────────────────────────────────────────────
                    if sim_time >= next_render:
                        if _cam_mode[0] == 1:
                            _qw, _qx, _qy, _qz = mujoco_go2.data.qpos[3:7]
                            _yaw = np.arctan2(
                                2.0 * (_qw * _qz + _qx * _qy),
                                1.0 - 2.0 * (_qy**2 + _qz**2),
                            )
                            viewer.cam.azimuth = np.degrees(_yaw)
                        viewer.sync()
                        next_render += RENDER_DT

                # ── Episode summary ────────────────────────────────────────
                final_dist = float(np.linalg.norm(_base_xy(mujoco_go2) - goal_xy))
                clean = goal_reached and not wall_touched
                status = ("CLEAN  " if clean
                          else "GOAL+W " if goal_reached
                          else "TIMEOUT" if (args.max_time is not None
                                             and float(mujoco_go2.data.time) >= args.max_time)
                          else "SKIPPED")
                print(f"\n  ep {ep:3d}  {status}  "
                      f"sim={float(mujoco_go2.data.time):6.2f}s  "
                      f"bumps={wall_contacts:3d}  dist={final_dist:.3f}m")

        except KeyboardInterrupt:
            print("\n\nCtrl-C — stopping.")
        finally:
            hud.stop()
            controller.stop()

    print(f"\nDemo finished after {ep} episode(s).")


if __name__ == "__main__":
    main()
