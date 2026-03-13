"""Glove controller for Go2 teleop via serial communication.

Adapted from reference/glove_controller.py.
Updated for united_0310.ino serial format:
  X_corr, Y_corr, Z_corr, Heading, Displacement,
  State, Zone, HapticMode, Bending:OXX, Fingers

Finger bitmask (from .ino):
  bit 0 (0x01) = Middle  finger [3]
  bit 1 (0x02) = Ring    finger [4]
  bit 2 (0x04) = Pinky   finger [5]

State machine mirrors united_0310.ino (magnitude-based, hysteresis):
  0 = OFF      (magnitude < THRESHOLD_OFF_MAX)
  1 = WAITING  (transitioning to ON, 500 ms timer)
  2 = ON       (magnitude >= THRESHOLD_ON_MIN)
  3 = HOVERING (THRESHOLD_HOVER_MIN <= magnitude < THRESHOLD_HOVER_MAX)
  Dead zone    (THRESHOLD_HOVER_MAX <= magnitude < THRESHOLD_ON_MIN) → hold state
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Finger combo bitmask constants  (match .ino defines)
# ---------------------------------------------------------------------------
COMBO_NONE              = 0x00  # 000 — idle / no fingers bent
COMBO_MIDDLE            = 0x01  # 001 — Middle only       [VR, other team]
COMBO_RING              = 0x02  # 010 — Ring only         [posture toggle]
COMBO_MIDDLE_RING       = 0x03  # 011 — Middle + Ring     [loco / recovery]
COMBO_RING_PINKY        = 0x06  # 110 — Ring  + Pinky     [arm, other team]
COMBO_MIDDLE_RING_PINKY = 0x07  # 111 — all three         [loco + euler]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GloveConfig:
    port: str = "/dev/ttyACM0"  # Linux; use COM3 / /dev/cu.usbmodem* on other OS
    baud_rate: int = 115200

    # Magnitude thresholds — match united_0310.ino constants exactly
    # Arduino uses 3D norm: sqrt(X_corr^2 + Y_corr^2 + Z_corr^2)
    thresh_off_max:   float = 500.0    # THRESHOLD_OFF_MAX   — fully away
    thresh_hover_min: float = 500.0    # THRESHOLD_HOVER_MIN — hover zone entry
    thresh_hover_max: float = 1300.0   # THRESHOLD_HOVER_MAX — hover zone exit
    thresh_on_min:    float = 1500.0   # THRESHOLD_ON_MIN    — magnet close
    # Dead zone: thresh_hover_max <= mag < thresh_on_min → hold previous state

    activation_time:  float = 0.5     # ACTIVATION_TIME_MS / 1000

    # Displacement scaling — match .ino constants
    max_displacement: float = 80.0    # MAX_DISPLACEMENT in .ino

    # Deadzone: displacements below this ratio of max are ignored
    deadzone_ratio: float = 0.35      # CENTER_DEADZONE / MAX_DISPLACEMENT ≈ 28/80

    # Velocity scaling
    max_lin_vel: float = 0.8    # m/s
    max_ang_vel: float = 1.5    # rad/s

    # "holonomic"    → full (vx, vy, 0)  strafing
    # "differential" → (vx, 0, wz)  like a wheeled robot
    control_mode: str = "holonomic"


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class GloveController:
    """Thread-safe glove input via serial.

    Background thread reads serial lines at ~1 kHz and updates internal
    state.  Main thread calls get_velocity_command() / get_finger_combo()
    at any rate.

    State machine mirrors united_0310.ino:
      OFF      → WAITING  (magnitude >= thresh_on_min, start 500 ms timer)
      WAITING  → ON       (held >= activation_time)
      ON/WAIT  → HOVERING (thresh_hover_min <= magnitude < thresh_hover_max)
      HOVERING → WAITING  (magnitude >= thresh_on_min, restart timer)
      any      → OFF      (magnitude < thresh_off_max)
      dead zone (thresh_hover_max <= mag < thresh_on_min) → hold current state
    """

    def __init__(self, config: GloveConfig | None = None):
        self.cfg = config or GloveConfig()

        self._serial = None
        self._serial_thread = None
        self._running = False
        self._lock = threading.Lock()

        # ── Raw sensor readings ──────────────────────────────────────────
        self._x_corr      = 0.0
        self._y_corr      = 0.0
        self._z_corr      = 0.0
        self._heading     = 0.0   # degrees
        self._displacement= 0.0   # sensor units
        self._zone        = 0     # 0–4 haptic zone
        self._arduino_state = 0   # 0=OFF,1=WAITING,2=ON,3=HOVERING from Arduino
        self._finger_bent = [False, False, False]   # Middle, Ring, Pinky
        self._finger_combo= 0     # bitmask

        # ── Python state machine ─────────────────────────────────────────
        # States: "OFF" | "WAITING" | "ON" | "HOVERING"
        self._ui_state            = "OFF"
        self._detection_start_time= 0.0

        # ── Velocity outputs ─────────────────────────────────────────────
        self._lin_x = 0.0
        self._lin_y = 0.0
        self._ang_z = 0.0

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open serial port and start background read thread."""
        try:
            import serial
            self._serial = serial.Serial(self.cfg.port, self.cfg.baud_rate, timeout=1)
            print(f"Glove connected on {self.cfg.port}")
        except Exception as e:
            print(f"Failed to connect glove: {e}")
            return False

        self._running = True
        self._serial_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._serial_thread.start()
        return True

    def stop(self) -> None:
        """Stop background thread and close serial port."""
        self._running = False
        if self._serial_thread is not None:
            self._serial_thread.join(timeout=1.0)
        if self._serial is not None:
            self._serial.close()

    def get_velocity_command(self) -> Tuple[float, float, float]:
        """Thread-safe read of (lin_x, lin_y, ang_z).  Returns (0,0,0) when not ON."""
        with self._lock:
            return self._lin_x, self._lin_y, self._ang_z

    def get_finger_combo(self) -> int:
        """Thread-safe read of finger bitmask (0x00–0x07)."""
        with self._lock:
            return self._finger_combo

    def get_state(self) -> str:
        """Returns "OFF", "WAITING", "ON", or "HOVERING"."""
        with self._lock:
            return self._ui_state

    def is_active(self) -> bool:
        with self._lock:
            return self._ui_state == "ON"

    def is_hovering(self) -> bool:
        """True when in HOVERING state (magnet in intermediate distance range).

        Use this to trigger a feature in the feature manager.
        Corresponds to Arduino uiState == 3 (HOVERING).
        """
        with self._lock:
            return self._ui_state == "HOVERING"

    def get_raw(self) -> dict:
        """Return all raw sensor values (useful for debugging)."""
        with self._lock:
            return {
                "x_corr":       self._x_corr,
                "y_corr":       self._y_corr,
                "z_corr":       self._z_corr,
                "heading":      self._heading,
                "displacement": self._displacement,
                "zone":         self._zone,
                "arduino_state":self._arduino_state,
                "finger_bent":  list(self._finger_bent),
                "finger_combo": self._finger_combo,
            }

    # ── Background thread ────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        while self._running:
            try:
                if self._serial and self._serial.in_waiting:
                    line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                    self._parse_line(line)
                    self._update_state()
            except Exception as e:
                print(f"\nSerial error: {e}")
                time.sleep(0.01)
            time.sleep(0.001)   # ~1 kHz poll

    def _parse_line(self, line: str) -> None:
        """Extract fields from one serial line into internal state."""
        m_x    = re.search(r"X_corr:([-0-9.]+)",      line)
        m_y    = re.search(r"Y_corr:([-0-9.]+)",      line)
        m_z    = re.search(r"Z_corr:([-0-9.]+)",      line)
        m_head = re.search(r"Heading:([-0-9.]+)",     line)
        m_disp = re.search(r"Displacement:([-0-9.]+)",line)
        m_stat = re.search(r"State:(\d+)",            line)
        m_zone = re.search(r"Zone:(\d+)",             line)
        m_bend = re.search(r"Bending:([OX]+)",        line)
        m_fing = re.search(r"Fingers:(\d+)",          line)

        with self._lock:
            if m_x:    self._x_corr       = float(m_x.group(1))
            if m_y:    self._y_corr       = float(m_y.group(1))
            if m_z:    self._z_corr       = float(m_z.group(1))
            if m_head: self._heading      = float(m_head.group(1))
            if m_disp: self._displacement = float(m_disp.group(1))
            if m_stat: self._arduino_state= int(m_stat.group(1))
            if m_zone: self._zone         = int(m_zone.group(1))
            if m_fing: self._finger_combo = int(m_fing.group(1))

            if m_bend:
                s = m_bend.group(1)
                for i in range(min(len(s), 3)):
                    # .ino: fingerBent[i] ? "O" : "X"  →  O means BENT
                    self._finger_bent[i] = (s[i] == "O")

    def _update_state(self) -> None:
        """Run Python-side state machine mirroring united_0310.ino.

        Uses 3D magnitude = sqrt(X_corr^2 + Y_corr^2 + Z_corr^2).

        Thresholds (matching .ino):
          magnitude >= thresh_on_min (1500)               → target ON
          thresh_hover_min (600) <= mag < thresh_hover_max (1300) → target HOVERING
          magnitude < thresh_off_max (600)                → target OFF
          thresh_hover_max (1300) <= mag < thresh_on_min (1500)   → dead zone, hold state

        Transitions:
          OFF/HOVERING + target=ON  → WAITING (start 500 ms timer)
          WAITING      + target=ON  → ON (after activation_time)
          ON/WAITING   + target=HOVERING → HOVERING (immediate)
          any          + target=OFF → OFF (immediate)
          dead zone                → no change
        """
        with self._lock:
            mag = float(np.sqrt(
                self._x_corr ** 2 + self._y_corr ** 2 + self._z_corr ** 2
            ))
            now = time.time()
            cfg = self.cfg

            # Determine target (None = dead zone, hold current state)
            if mag >= cfg.thresh_on_min:
                target = "ON"
            elif cfg.thresh_hover_min <= mag < cfg.thresh_hover_max:
                target = "HOVERING"
            elif mag < cfg.thresh_off_max:
                target = "OFF"
            else:
                target = None   # dead zone: thresh_hover_max <= mag < thresh_on_min

            if target == "ON":
                if self._ui_state in ("OFF", "HOVERING"):
                    self._ui_state = "WAITING"
                    self._detection_start_time = now
                elif self._ui_state == "WAITING":
                    if now - self._detection_start_time >= cfg.activation_time:
                        self._ui_state = "ON"
                # ON: stay ON

            elif target == "HOVERING":
                # Immediate transition to HOVERING from any active state
                self._ui_state = "HOVERING"
                self._detection_start_time = 0.0

            elif target == "OFF":
                self._ui_state = "OFF"
                self._detection_start_time = 0.0

            # target is None (dead zone) → no state change

            # Velocity output
            if self._ui_state == "ON":
                self._compute_velocity()
            else:
                self._lin_x = self._lin_y = self._ang_z = 0.0

    def _compute_velocity(self) -> None:
        """Convert heading + displacement → (lin_x, lin_y, ang_z).  Lock held by caller."""
        deadzone = self.cfg.max_displacement * self.cfg.deadzone_ratio
        if self._displacement < deadzone:
            self._lin_x = self._lin_y = self._ang_z = 0.0
            return

        norm = min(self._displacement, self.cfg.max_displacement) / self.cfg.max_displacement
        rad  = np.deg2rad(self._heading)

        if self.cfg.control_mode == "holonomic":
            # heading 0° = forward, 90° CCW = left
            self._lin_x = norm * self.cfg.max_lin_vel * np.cos(rad)
            self._lin_y = norm * self.cfg.max_lin_vel * np.sin(rad)
            self._ang_z = 0.0
        else:   # differential
            self._lin_x = norm * self.cfg.max_lin_vel * np.cos(rad)
            self._lin_y = 0.0
            self._ang_z = norm * self.cfg.max_ang_vel * np.sin(rad)
