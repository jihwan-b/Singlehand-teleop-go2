"""Quest 3 controller input via OpenVR / SteamVR.

Setup
-----
  1. Start SteamVR on PC.
  2. Connect Quest 3 via ALVR (it will appear as a tracked headset in SteamVR).
  3. pip install openvr   (pyopenvr package)
  4. Run run_quest3.py — no extra arguments needed.

Presents the same public API as GloveController so the feature manager and
MPC loop work without modification.

Button → Combo mapping  (mirrors glove finger-bitmask convention):
  Right trigger held         →  state "ON"  + combo 0x03  MR-  (Locomotion)
  Right trigger + grip held  →  state "ON"  + combo 0x07  MRP  (Loco + Euler tilt)
  B (right) or Y (left) tap  →  state "OFF" + combo 0x02  -R-  (Posture toggle)
  Nothing / trigger released →  state "OFF" + combo 0x00  ---  (IDLE)

Velocity mapping  (body-frame, no auto-yaw needed):
  Left  stick Y  →  +vx  (forward / back)
  Left  stick X  →  -vy  (strafe right / left; note sign: +Y = left in robot frame)
  Right stick X  →  -wz  (yaw right / left; CCW positive in std robotics)

Unlike the glove, Quest 3 does not have a thumb-push-away (DAMP) gesture.
is_thumb_away() always returns False.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QuestConfig:
    max_lin_vel: float = 0.8    # m/s
    max_ang_vel: float = 1.5    # rad/s
    trigger_threshold: float = 0.5   # 0–1; trigger must exceed this to go "ON"
    deadzone: float = 0.12           # stick deadzone (fraction of full range)


class QuestController:
    """Thread-safe Quest 3 controller via OpenVR.

    Background thread polls SteamVR at ~200 Hz and updates internal state.
    Main thread calls get_velocity_command() / get_finger_combo() at any rate.
    """

    def __init__(self, config: QuestConfig | None = None):
        self.cfg = config or QuestConfig()

        self._vr      = None
        self._openvr  = None
        self._thread  = None
        self._running = False
        self._lock    = threading.Lock()

        self._vx    = 0.0
        self._vy    = 0.0
        self._wz    = 0.0
        self._combo = 0
        self._state = "OFF"

    # ── Public API (mirrors GloveController) ─────────────────────────────────

    def start(self) -> bool:
        """Initialise OpenVR and start background poll thread."""
        try:
            import openvr
            self._openvr = openvr
            self._vr = openvr.init(openvr.VRApplication_Background)
            print("Quest 3 connected via OpenVR / SteamVR")
        except Exception as e:
            print(f"Failed to connect Quest 3 via OpenVR: {e}")
            print("  → Is SteamVR running?  Is 'openvr' installed (pip install openvr)?")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop background thread and shut down OpenVR."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._vr is not None:
            try:
                import openvr
                openvr.shutdown()
            except Exception:
                pass

    def get_velocity_command(self) -> Tuple[float, float, float]:
        """Thread-safe (vx, vy, wz) in robot body frame.  (0, 0, 0) when OFF."""
        with self._lock:
            return self._vx, self._vy, self._wz

    def get_finger_combo(self) -> int:
        """Thread-safe finger-combo bitmask (0x00–0x07)."""
        with self._lock:
            return self._combo

    def get_state(self) -> str:
        """Returns 'OFF' or 'ON'."""
        with self._lock:
            return self._state

    def is_active(self) -> bool:
        with self._lock:
            return self._state == "ON"

    def is_thumb_away(self) -> bool:
        """Always False — Quest 3 has no DAMP equivalent."""
        return False

    def get_raw(self) -> dict:
        """Diagnostic: return last polled values."""
        with self._lock:
            return {
                "vx":    self._vx,
                "vy":    self._vy,
                "wz":    self._wz,
                "combo": self._combo,
                "state": self._state,
            }

    # ── Background thread ─────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        while self._running:
            try:
                self._poll()
            except Exception as e:
                print(f"\nOpenVR poll error: {e}")
            time.sleep(0.005)   # ~200 Hz

    def _poll(self) -> None:
        openvr = self._openvr
        vr     = self._vr

        # Find left and right controller device indices
        left_idx  = openvr.k_unTrackedDeviceIndexInvalid
        right_idx = openvr.k_unTrackedDeviceIndexInvalid

        for i in range(1, openvr.k_unMaxTrackedDeviceCount):
            if vr.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_Controller:
                continue
            role = vr.getControllerRoleForTrackedDeviceIndex(i)
            if role == openvr.TrackedControllerRole_LeftHand:
                left_idx = i
            elif role == openvr.TrackedControllerRole_RightHand:
                right_idx = i

        # If right controller is not found, go to OFF
        if right_idx == openvr.k_unTrackedDeviceIndexInvalid:
            with self._lock:
                self._state = "OFF"
                self._vx = self._vy = self._wz = 0.0
                self._combo = 0
            return

        _, right_st = vr.getControllerState(right_idx)
        left_st = None
        if left_idx != openvr.k_unTrackedDeviceIndexInvalid:
            _, left_st = vr.getControllerState(left_idx)

        # ── Right controller ──────────────────────────────────────────────────
        rbtn = right_st.ulButtonPressed

        # Trigger value (axis 1 x-component, range 0–1)
        trigger_val = right_st.rAxis[1].x
        active      = trigger_val >= self.cfg.trigger_threshold

        # Grip button
        grip_held = bool(rbtn & (1 << openvr.k_EButton_Grip))

        # B button on right controller (ApplicationMenu in OpenVR)
        b_pressed = bool(rbtn & (1 << openvr.k_EButton_ApplicationMenu))

        # Right thumbstick X → yaw  (axis 0)
        right_stick_x = right_st.rAxis[0].x

        # ── Left controller ───────────────────────────────────────────────────
        y_pressed  = False
        left_stick_x = 0.0
        left_stick_y = 0.0

        if left_st is not None:
            lbtn = left_st.ulButtonPressed
            # Y button on left controller (also ApplicationMenu)
            y_pressed    = bool(lbtn & (1 << openvr.k_EButton_ApplicationMenu))
            left_stick_x = left_st.rAxis[0].x
            left_stick_y = left_st.rAxis[0].y

        # ── Deadzone ──────────────────────────────────────────────────────────
        dz = self.cfg.deadzone

        def _dz(v: float) -> float:
            return 0.0 if abs(v) < dz else v

        raw_vx = _dz(left_stick_y)    # forward = stick up = positive
        raw_vy = _dz(-left_stick_x)   # left    = stick left = positive (robot +Y = left)
        raw_wz = _dz(-right_stick_x)  # CCW     = stick left = positive (std robotics)

        # ── Scale ─────────────────────────────────────────────────────────────
        vx = raw_vx * self.cfg.max_lin_vel
        vy = raw_vy * self.cfg.max_lin_vel
        wz = raw_wz * self.cfg.max_ang_vel

        # ── Combo (mirrors glove bitmask) ─────────────────────────────────────
        if active and grip_held:
            combo = 0x07   # MRP — locomotion + Euler tilt
        elif active:
            combo = 0x03   # MR- — locomotion
        elif b_pressed or y_pressed:
            combo = 0x02   # -R- — posture toggle (single tap)
        else:
            combo = 0x00   # --- — idle

        state = "ON" if active else "OFF"

        if not active:
            vx = vy = wz = 0.0

        with self._lock:
            self._state = state
            self._vx    = vx
            self._vy    = vy
            self._wz    = wz
            self._combo = combo
