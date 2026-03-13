"""Go2 Glove Teleoperation (body-frame joystick mode)  —  real-time MuJoCo simulation.

Glove magnetometer heading is interpreted in body frame (like a joystick),
not as an absolute compass direction.  No auto-yaw.

Usage
-----
    python teleop/run_teleop_new.py
    python teleop/run_teleop_new.py --port /dev/ttyACM0
    python teleop/run_teleop_new.py --maxv 0.6 --maxw 1.2
    python teleop/run_teleop_new.py --no-hud

Controls (finger combos)
------------------------
  Middle+Ring held (MR)        →  Holonomic locomotion
                                   Glove tilt → body-frame (vx, vy), no yaw change
                                   Like Quest3 left stick only
  Middle+Ring+Pinky held (MRP) →  Differential locomotion + yaw
                                   Glove forward → vx,  Glove sideways → ang_z
                                   Like Quest3 left + right stick (single input)
  Ring tap (alone)             →  Posture toggle  StandUp ↔ StandDown
  Middle+Ring  ×2 quick        →  RecoveryStand
  No fingers / glove OFF       →  Robot trots in place (zero velocity)

Camera (press T in viewer)
--------------------------
  Mode 1  →  3rd-person follow (behind robot, azimuth tracks yaw)
  Mode 2  →  1st-person (Go2 head camera)
  Mode 0  →  Free camera (manual)

Glove activation
----------------
  Hold the magnet near the sensor for 0.5 s.
  Remove the magnet → robot stops immediately.

Exit
----
  Close the MuJoCo viewer window  OR  Ctrl-C in terminal.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

# ── Add project root to path so both convex_mpc and teleop are importable ────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model   import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait            import Gait

from teleop.glove_controller import GloveController, GloveConfig
from teleop.feature_trigger  import RobotFeatureManager
from teleop.hud              import TeleopState, TeleopHUD


# ── Rate / timing constants ──────────────────────────────────────────────────

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

STATUS_EVERY = 2


# ── Joint torque limits ──────────────────────────────────────────────────────

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


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    glove_state: str,
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
        f"Glove:{glove_state:8s}  [{lbl}]  "
        f"vx:{vx:+.2f} vy:{vy:+.2f} wz:{wz:+.2f}  "
        f"{feat_str}   ",
        end="",
        flush=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 glove teleoperation (body-frame joystick)")
    parser.add_argument("--port",  default="/dev/ttyACM0",
                        help="Serial port (default: /dev/ttyACM0)")
    parser.add_argument("--maxv",  type=float, default=0.8,
                        help="Max linear velocity in m/s (default: 0.8)")
    parser.add_argument("--maxw",  type=float, default=1.5,
                        help="Max angular velocity in rad/s (default: 1.5)")
    parser.add_argument("--no-hud", dest="hud", action="store_false",
                        help="Disable live matplotlib HUD panel")
    parser.set_defaults(hud=True)
    args = parser.parse_args()

    # ── Robot initialisation ─────────────────────────────────────────────
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

    # ── Glove + feature manager ──────────────────────────────────────────
    glove_cfg = GloveConfig(
        port=args.port,
        max_lin_vel=args.maxv,
        max_ang_vel=args.maxw,
    )
    glove = GloveController(glove_cfg)
    feat  = RobotFeatureManager(ctrl_dt=CTRL_DT)

    connected = glove.start()
    if not connected:
        print("WARNING: glove not connected — robot will stand with zero commands")

    # ── HUD (optional) ───────────────────────────────────────────────────
    hud_state: TeleopState | None = None
    if args.hud:
        hud_state = TeleopState()
        TeleopHUD(hud_state).start()
        print("HUD started (separate window)")

    # ── Banner ───────────────────────────────────────────────────────────
    print("─" * 70)
    print("Go2 Glove Teleop (body-frame joystick)  |  close viewer or Ctrl-C to stop")
    print(f"  Port: {args.port}  MaxV: {args.maxv} m/s  MaxW: {args.maxw} rad/s")
    print("  MR-  = Holonomic (vx+vy, no yaw)  |  MRP = Differential (vx+wz)")
    print("  -R-  tap = Posture  |  MR- ×2 = Recovery")
    print("  Press T in viewer to cycle camera: 3rd-person → 1st-person → Free")
    print("─" * 70)

    # ── Loop state ───────────────────────────────────────────────────────
    k             = 0
    ctrl_i        = 0
    tau_hold      = np.zeros(12, dtype=float)
    next_render_t = 0.0
    wall_start    = time.perf_counter()

    # ── Camera mode cycle (press T in viewer window) ─────────────────────
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
        else:  # FREE
            viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE

    def _key_callback(keycode: int) -> None:
        if keycode != GLFW_KEY_T:
            return
        _cam_mode[0] = (_cam_mode[0] + 1) % 3
        _apply_cam_mode(_cam_mode[0])
        print(f"\nCamera: {_CAM_LABELS[_cam_mode[0]]}")

    # ── Real-time loop ───────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        mujoco_go2.model,
        mujoco_go2.data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=_key_callback,
    ) as viewer:
        _apply_cam_mode(_cam_mode[0])

        try:
            while viewer.is_running():
                sim_time = float(mujoco_go2.data.time)

                # ── Real-time pacing ─────────────────────────────────────
                wall_elapsed = time.perf_counter() - wall_start
                ahead = sim_time - wall_elapsed
                if ahead > 0.0:
                    time.sleep(ahead)

                # ── Control at CTRL_HZ  (every CTRL_DECIM physics steps) ─
                if k % CTRL_DECIM == 0:

                    # Read glove
                    combo       = glove.get_finger_combo()
                    raw_vx, raw_vy, raw_wz = glove.get_velocity_command()
                    glove_state = glove.get_state()
                    hovering    = glove.is_hovering()

                    # Feature manager → final commands
                    cmd   = feat.update(combo, raw_vx, raw_vy, raw_wz,
                                        hovering, sim_time)
                    x_vel = cmd["x_vel"]   # world-frame from glove magnetometer
                    y_vel = cmd["y_vel"]
                    z_pos = cmd["z_pos"]
                    euler_shift = cmd["euler_shift"]  # True when combo = MRP (0x07)

                    # ── Sync Pinocchio from MuJoCo ───────────────────────
                    mujoco_go2.update_pin_with_mujoco(go2)

                    # ── Body-frame velocity ──────────────────────────────
                    # Glove compass heading → used directly as body-frame command.
                    # Same locomotion logic for all camera modes.
                    x_vel_b  = x_vel
                    y_vel_b  = y_vel
                    _speed_b = np.hypot(x_vel_b, y_vel_b)

                    # ── Map to MPC commands ──────────────────────────────
                    if euler_shift:
                        # MRP (0x07): differential — forward + yaw from single input.
                        # Side component → ang_z (turn rate); no strafe.
                        ang_z   = float(np.clip(
                            glove_cfg.max_ang_vel * (y_vel_b / _speed_b if _speed_b > 0.01 else 0.0),
                            -glove_cfg.max_ang_vel,
                            glove_cfg.max_ang_vel,
                        ))
                        y_vel_b = 0.0
                    else:
                        # MR (0x03): holonomic — move in body-frame direction, no yaw.
                        ang_z = 0.0

                    # ── MPC solve  (every STEPS_PER_MPC control ticks) ───
                    if ctrl_i % STEPS_PER_MPC == 0:
                        traj.generate_traj(
                            go2, gait, sim_time,
                            x_vel_b, y_vel_b, z_pos, ang_z,
                            time_step=MPC_DT,
                        )

                        sol   = mpc.solve_QP(go2, traj, False)
                        N     = traj.N
                        w_opt = sol["x"].full().flatten()
                        U_opt = w_opt[12 * N :].reshape((12, N), order="F")

                        _print_status(
                            sim_time, glove_state, combo,
                            x_vel_b, y_vel_b, ang_z,
                            feat.status_str(),
                        )

                        if hud_state is not None:
                            hud_state.update(
                                sim_time=sim_time,
                                glove_state=glove_state,
                                combo=combo,
                                vx=x_vel_b,
                                vy=y_vel_b,
                                wz=ang_z,
                                status=feat.status_str(),
                                z_pos=z_pos,
                            )

                    mpc_force = U_opt[:, 0]

                    # ── Leg torques ──────────────────────────────────────
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

                # ── Physics step ─────────────────────────────────────────
                mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
                mujoco_go2.set_joint_torque(tau_hold)
                mj.mj_step2(mujoco_go2.model, mujoco_go2.data)
                k += 1

                # ── Viewer sync at RENDER_HZ ─────────────────────────────
                if float(mujoco_go2.data.time) >= next_render_t:
                    # 3rd-person: rotate azimuth to always stay behind robot
                    if _cam_mode[0] == 1:
                        _qw, _qx, _qy, _qz = mujoco_go2.data.qpos[3:7]
                        _yaw = np.arctan2(
                            2.0 * (_qw * _qz + _qx * _qy),
                            1.0 - 2.0 * (_qy ** 2 + _qz ** 2),
                        )
                        viewer.cam.azimuth = np.degrees(_yaw)
                    viewer.sync()
                    next_render_t += RENDER_DT

        except KeyboardInterrupt:
            print("\n\nCtrl-C — stopping...")

        finally:
            glove.stop()
            print("Teleop session ended.")


if __name__ == "__main__":
    main()
