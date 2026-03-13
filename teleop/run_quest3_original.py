"""Go2 Quest 3 Teleoperation  —  real-time MuJoCo simulation.

Requires:
  - SteamVR running with ALVR streaming Quest 3
  - pip install openvr

Usage
-----
    python teleop/run_quest3.py
    python teleop/run_quest3.py --mode differential
    python teleop/run_quest3.py --maxv 0.6 --maxw 1.2
    python teleop/run_quest3.py --no-hud

Controls
--------
  Right trigger (held)      →  Locomotion  (left stick = forward/strafe, right stick = yaw)
  Right trigger + grip      →  Locomotion + Euler tilt  (lean toward movement)
  B (right) / Y (left) tap  →  Posture toggle  StandUp ↔ StandDown
  Release trigger            →  Robot stops (zero velocity)

Velocity axes  (body frame — no auto-yaw, you steer with right stick):
  Left  stick ↑ / ↓   →  forward / back
  Left  stick → / ←   →  strafe right / left
  Right stick → / ←   →  yaw right / left

Camera toggle: press T in the MuJoCo viewer window.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

# ── Add project root to path ──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model   import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait            import Gait

from teleop.quest3_controller import QuestController, QuestConfig
from teleop.feature_trigger   import RobotFeatureManager
from teleop.hud               import TeleopState, TeleopHUD


# ── Rate / timing constants ───────────────────────────────────────────────────

GAIT_HZ    = 3
GAIT_DUTY  = 0.6
GAIT_T     = 1.0 / GAIT_HZ

SIM_HZ     = 1000
SIM_DT     = 1.0 / SIM_HZ

CTRL_HZ    = 200
CTRL_DT    = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ          # = 5 physics steps per control tick

MPC_DT        = GAIT_T / 16             # ≈ 20.8 ms
MPC_HZ        = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))   # ≈ 4 control ticks per MPC

RENDER_HZ  = 60.0
RENDER_DT  = 1.0 / RENDER_HZ

STATUS_EVERY = 2   # print every N MPC solves


# ── Joint torque limits ───────────────────────────────────────────────────────

_SAFETY = 0.9
TAU_LIM = _SAFETY * np.array([
    23.7, 23.7, 45.43,   # FL: hip, thigh, calf
    23.7, 23.7, 45.43,   # FR
    23.7, 23.7, 45.43,   # RL
    23.7, 23.7, 45.43,   # RR
])

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

_COMBO_LABEL = {
    0x00: "---",
    0x01: "M--",
    0x02: "-R-",
    0x03: "MR-",
    0x04: "--P",
    0x06: "-RP",
    0x07: "MRP",
}

_status_cnt = 0

def _print_status(
    sim_time: float,
    ctrl_state: str,
    combo: int,
    vx: float,
    vy: float,
    wz: float,
    feat_str: str,
) -> None:
    global _status_cnt
    _status_cnt += 1
    if _status_cnt % STATUS_EVERY != 0:
        return
    lbl = _COMBO_LABEL.get(combo, f"{combo:03b}")
    print(
        f"\rt={sim_time:6.1f}s  "
        f"Quest:{ctrl_state:4s}  [{lbl}]  "
        f"vx:{vx:+.2f} vy:{vy:+.2f} wz:{wz:+.2f}  "
        f"{feat_str}   ",
        end="",
        flush=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 Quest 3 teleoperation")
    parser.add_argument("--mode", default="holonomic",
                        choices=["holonomic", "differential"],
                        help="Control mode (default: holonomic)")
    parser.add_argument("--maxv", type=float, default=0.8,
                        help="Max linear velocity in m/s (default: 0.8)")
    parser.add_argument("--maxw", type=float, default=1.5,
                        help="Max angular velocity in rad/s (default: 1.5)")
    parser.add_argument("--no-hud", dest="hud", action="store_false",
                        help="Disable live matplotlib HUD panel")
    parser.set_defaults(hud=True)
    args = parser.parse_args()

    # ── Robot initialisation ──────────────────────────────────────────────
    go2        = PinGo2Model()
    mujoco_go2 = MuJoCo_GO2_Model()
    leg_ctrl   = LegController()
    traj       = ComTraj(go2)
    gait       = Gait(GAIT_HZ, GAIT_DUTY)

    q_init = go2.current_config.get_q()
    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = SIM_DT

    traj.generate_traj(go2, gait, 0.0, 0.0, 0.0, 0.27, 0.0, time_step=MPC_DT)
    mpc   = CentroidalMPC(go2, traj)
    U_opt = np.zeros((12, traj.N), dtype=float)

    # ── Quest 3 + feature manager ─────────────────────────────────────────
    quest_cfg = QuestConfig(
        max_lin_vel=args.maxv,
        max_ang_vel=args.maxw,
    )
    quest = QuestController(quest_cfg)
    feat  = RobotFeatureManager(ctrl_dt=CTRL_DT)

    connected = quest.start()
    if not connected:
        print("WARNING: Quest 3 not connected — robot will stand with zero commands")

    # ── HUD (optional) ────────────────────────────────────────────────────
    hud_state: TeleopState | None = None
    if args.hud:
        hud_state = TeleopState()
        TeleopHUD(hud_state).start()
        print("HUD started (separate window)")

    # ── Banner ────────────────────────────────────────────────────────────
    print("─" * 70)
    print("Go2 Quest 3 Teleop  |  close viewer window or Ctrl-C to stop")
    print(f"  Mode: {args.mode}  MaxV: {args.maxv} m/s  MaxW: {args.maxw} rad/s")
    print("  Right trigger = Locomotion  |  + grip = Loco+Euler  |  B/Y tap = Posture")
    print("  Left stick = forward/strafe  |  Right stick = yaw")
    print("  Press T in viewer to toggle camera follow mode")
    print("─" * 70)

    # ── Loop state ────────────────────────────────────────────────────────
    k             = 0
    ctrl_i        = 0
    tau_hold      = np.zeros(12, dtype=float)
    next_render_t = 0.0
    wall_start    = time.perf_counter()

    # ── Camera follow toggle (T key) ──────────────────────────────────────
    _cam_follow = [True]
    GLFW_KEY_T  = 84

    def _key_callback(keycode: int) -> None:
        if keycode != GLFW_KEY_T:
            return
        _cam_follow[0] = not _cam_follow[0]
        if _cam_follow[0]:
            viewer.cam.type        = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco_go2.model.body("base_link").id
        else:
            viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
        print(f"\nCamera: {'TRACKING (follow)' if _cam_follow[0] else 'FREE (manual)'}")

    # ── Real-time loop ────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        mujoco_go2.model,
        mujoco_go2.data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=_key_callback,
    ) as viewer:
        viewer.cam.type        = mj.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco_go2.model.body("base_link").id
        viewer.cam.distance    = 3.0
        viewer.cam.azimuth     = 180
        viewer.cam.elevation   = -20

        try:
            while viewer.is_running():
                sim_time = float(mujoco_go2.data.time)

                # Real-time pacing
                wall_elapsed = time.perf_counter() - wall_start
                ahead = sim_time - wall_elapsed
                if ahead > 0.0:
                    time.sleep(ahead)

                # Control at CTRL_HZ  (every CTRL_DECIM physics steps)
                if k % CTRL_DECIM == 0:

                    # Read Quest 3
                    combo       = quest.get_finger_combo()
                    raw_vx, raw_vy, raw_wz = quest.get_velocity_command()
                    ctrl_state  = quest.get_state()
                    # Feature manager → final commands
                    cmd   = feat.update(combo, raw_vx, raw_vy, raw_wz,
                                        False, sim_time)
                    x_vel = cmd["x_vel"]
                    y_vel = cmd["y_vel"]
                    ang_z = cmd["ang_z"]
                    z_pos = cmd["z_pos"]
                    euler_shift = cmd["euler_shift"]

                    # NOTE: No auto-yaw here.
                    # Quest 3 outputs body-frame velocity directly (left stick = robot
                    # forward/strafe, right stick = yaw).  The robot turns where you
                    # point it — you are in full manual control of heading.

                    # Sync Pinocchio from MuJoCo
                    mujoco_go2.update_pin_with_mujoco(go2)

                    # MPC solve  (every STEPS_PER_MPC control ticks)
                    if ctrl_i % STEPS_PER_MPC == 0:
                        traj.generate_traj(
                            go2, gait, sim_time,
                            x_vel, y_vel, z_pos, ang_z,
                            time_step=MPC_DT,
                        )

                        if euler_shift:
                            EULER_LEAN_GAIN = 0.12
                            MAX_LEAN_RAD    = 0.20
                            traj.rpy_traj_world[1, :] = np.clip(-EULER_LEAN_GAIN * x_vel,
                                                                 -MAX_LEAN_RAD, MAX_LEAN_RAD)
                            traj.rpy_traj_world[0, :] = np.clip(-EULER_LEAN_GAIN * y_vel,
                                                                 -MAX_LEAN_RAD, MAX_LEAN_RAD)

                        sol   = mpc.solve_QP(go2, traj, False)
                        N     = traj.N
                        w_opt = sol["x"].full().flatten()
                        U_opt = w_opt[12 * N :].reshape((12, N), order="F")

                        _print_status(
                            sim_time, ctrl_state, combo,
                            x_vel, y_vel, ang_z,
                            feat.status_str(),
                        )

                        if hud_state is not None:
                            hud_state.update(
                                sim_time=sim_time,
                                glove_state=ctrl_state,
                                combo=combo,
                                vx=x_vel,
                                vy=y_vel,
                                wz=ang_z,
                                status=feat.status_str(),
                                z_pos=z_pos,
                            )

                    mpc_force = U_opt[:, 0]

                    # Leg torques
                    tau_raw = np.zeros(12)
                    for leg in ("FL", "FR", "RL", "RR"):
                        out = leg_ctrl.compute_leg_torque(
                            leg, go2, gait,
                            mpc_force[LEG_SLICE[leg]],
                            sim_time,
                        )
                        tau_raw[LEG_SLICE[leg]] = out.tau

                    tau_hold = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
                    ctrl_i  += 1

                # Physics step
                mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
                mujoco_go2.set_joint_torque(tau_hold)
                mj.mj_step2(mujoco_go2.model, mujoco_go2.data)
                k += 1

                # Viewer sync at RENDER_HZ
                if float(mujoco_go2.data.time) >= next_render_t:
                    viewer.sync()
                    next_render_t += RENDER_DT

        except KeyboardInterrupt:
            print("\n\nCtrl-C — stopping...")

        finally:
            quest.stop()
            print("Quest 3 teleop session ended.")


if __name__ == "__main__":
    main()