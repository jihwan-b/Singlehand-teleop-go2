"""Glove controller for Go2 teleop via serial communication.

Adapted from reference/glove_controller.py.
Updated for united_0210.ino serial format:
  X_corr, Y_corr, Z_corr, Heading, Displacement,
  State, Zone, HapticMode, Bending:OXX, Fingers

Finger bitmask (from .ino):
  bit 0 (0x01) = Middle  finger [3]
  bit 1 (0x02) = Ring    finger [4]
  bit 2 (0x04) = Pinky   finger [5]
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

    # Sensor calibration (must match .ino constants)
    max_displacement: float = 80.0   # MAX_DISPLACEMENT in .ino
    magnet_threshold: float = 90.0   # MAGNET_THRESHOLD in .ino
    activation_time:  float = 0.5    # ACTIVATION_TIME_MS / 1000

    # Deadzone: displacements below this ratio of max are ignored
    deadzone_ratio: float = 0.35     # CENTER_DEADZONE / MAX_DISPLACEMENT ≈ 28/80

    # Velocity scaling
    max_lin_vel: float = 0.8    # m/s
    max_ang_vel: float = 1.5    # rad/s

    # "holonomic"   → full (vx, vy, 0)  strafing
    # "differential" → (vx, 0, wz)  like a wheeled robot
    control_mode: str = "holonomic"

    # Thumb "push away" gesture (엄지 멀리 때기)
    # When DAMP_LOW ≤ abs(Z_corr) < magnet_threshold → "DAMP" state
    # Physics: field ∝ 1/r³, so this zone is ~20-60% farther than ON position
    # Set to 0.0 to disable the DAMP state entirely.
    thumb_damp_low: float = 25.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class GloveController:
    """Thread-safe glove input via serial.

    Background thread reads serial lines at ~1 kHz and updates internal
    state.  Main thread calls get_velocity_command() / get_finger_combo()
    at any rate.

    State machine mirrors the Arduino:
      OFF  → WAITING (magnet detected)
      WAITING → ON   (held ≥ activation_time)
      * → OFF        (magnet removed)
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
        self._arduino_state = 0   # 0=OFF,1=WAITING,2=ON from Arduino
        self._finger_bent = [False, False, False]   # Middle, Ring, Pinky
        self._finger_combo= 0     # bitmask

        # ── Python state machine ─────────────────────────────────────────
        # States: "OFF" | "WAITING" | "ON" | "DAMP"
        #   DAMP = thumb partially pulled away (DAMP_LOW ≤ Z_corr < threshold)
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
        """Thread-safe read of (lin_x, lin_y, ang_z).  Returns (0,0,0) when OFF."""
        with self._lock:
            return self._lin_x, self._lin_y, self._ang_z

    def get_finger_combo(self) -> int:
        """Thread-safe read of finger bitmask (0x00–0x07)."""
        with self._lock:
            return self._finger_combo

    def get_state(self) -> str:
        """Returns "OFF", "WAITING", or "ON"."""
        with self._lock:
            return self._ui_state

    def is_active(self) -> bool:
        with self._lock:
            return self._ui_state == "ON"

    def is_thumb_away(self) -> bool:
        """True when in DAMP state (thumb partially pulled away from sensor).

        Use this to trigger velocity-hold or exponential decay in the feature
        manager instead of an abrupt stop.
        """
        with self._lock:
            return self._ui_state == "DAMP"

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
        """Run Python-side state machine and recompute velocity output.

        State transitions:
          OFF     → (Z_corr ≥ threshold)               → WAITING
          WAITING → (held ≥ activation_time)            → ON
          ON      → (damp_low ≤ Z_corr < threshold)    → DAMP  (thumb away)
          DAMP    → (Z_corr ≥ threshold)                → ON    (thumb back)
          ON/DAMP → (Z_corr < damp_low)                 → OFF
          WAITING → (Z_corr < threshold)                → OFF
        """
        with self._lock:
            z_abs      = abs(self._z_corr)
            threshold  = self.cfg.magnet_threshold
            damp_low   = self.cfg.thumb_damp_low
            now        = time.time()

            is_magnet  = z_abs >= threshold
            is_damp    = (damp_low > 0.0) and (damp_low <= z_abs < threshold)
            is_gone    = z_abs < max(damp_low, 1.0)   # fully away

            if is_magnet:
                if self._ui_state in ("OFF", "DAMP"):
                    self._ui_state = "WAITING"
                    self._detection_start_time = now
                elif self._ui_state == "WAITING":
                    if now - self._detection_start_time >= self.cfg.activation_time:
                        self._ui_state = "ON"
                # ON: stay ON

            elif is_damp:
                # Thumb partially away — only meaningful when previously active
                if self._ui_state in ("ON",):
                    self._ui_state = "DAMP"
                elif self._ui_state == "WAITING":
                    self._ui_state = "OFF"   # never fully activated

            else:   # is_gone (or damp disabled and not in magnet range)
                self._ui_state = "OFF"
                self._detection_start_time = 0.0

            # Velocity is only recomputed here for ON; DAMP is handled externally
            if self._ui_state == "ON":
                self._compute_velocity()
            elif self._ui_state != "DAMP":
                # OFF / WAITING → zero velocity
                self._lin_x = self._lin_y = self._ang_z = 0.0
            # DAMP: leave _lin_x/y/z_ang unchanged (feature manager handles decay)

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
