"""Feature trigger primitives and robot feature manager for glove teleop.

Three trigger types
-------------------
ToggleTrigger    — flip a boolean on each rising edge (single tap)
LongPressTrigger — fire once after held for N seconds
DoubleTapTrigger — fire on rapid press-release-press (with max_hold guard
                   so a long locomotion hold does not trigger this)

Feature mapping  (fingerCombo bitmask, see glove_controller.py)
---------------------------------------------------------------
0x00 (none)     IDLE   — robot stands still (zero velocity)
0x01 (M)        [VR viewpoint — other team, ignored here]
0x02 (R)        Posture toggle — StandUp ↔ StandDown on each Ring tap
0x03 (M+R)      HOLD   → Locomotion (follow heading/displacement)
                x2 tap → RecoveryStand (zero vel, reset to stand height)
0x06 (R+P)      [Arm manipulation — other team, ignored here]
0x07 (M+R+P)    Locomotion + Euler tilt

Glove HOVERING state (magnet at intermediate distance, united_0310.ino state 3)
-------------------------------------------------------------------------------
Exposed via cmd["hovering"] = True for the main loop to act on.
Currently yields zero velocity; add specific hover-triggered behaviour here.
"""

from __future__ import annotations

from .glove_controller import (
    COMBO_NONE,
    COMBO_MIDDLE,
    COMBO_RING,
    COMBO_MIDDLE_RING,
    COMBO_RING_PINKY,
    COMBO_MIDDLE_RING_PINKY,
)

# Combos that enable locomotion
_LOCO_COMBOS: frozenset[int] = frozenset({COMBO_MIDDLE_RING, COMBO_MIDDLE_RING_PINKY})


# ── Primitive triggers ───────────────────────────────────────────────────────

class ToggleTrigger:
    """Flip internal state on each rising edge.

    update() returns True only on the tick where the state just changed.
    Read .state for the current latched value.

    Usage (posture toggle)::

        trig = ToggleTrigger()
        if trig.update(combo == COMBO_RING):
            is_crouched = trig.state   # True = crouched, False = standing
    """

    def __init__(self, initial: bool = False) -> None:
        self._state = initial
        self._prev  = False

    def update(self, pressed: bool) -> bool:
        """Call every control tick.  Returns True if state just flipped."""
        fired = pressed and not self._prev
        if fired:
            self._state = not self._state
        self._prev = pressed
        return fired

    @property
    def state(self) -> bool:
        return self._state


class LongPressTrigger:
    """Fire once after the input has been held for `hold_time` seconds.

    Resets when the input is released.  Will NOT re-fire while still held.

    Usage (e.g. mode switch after deliberate hold)::

        trig = LongPressTrigger(hold_time=0.8)
        if trig.update(is_pressed, sim_time):
            activate_mode()
    """

    def __init__(self, hold_time: float = 0.8) -> None:
        self.hold_time   = hold_time
        self._press_start: float | None = None
        self._fired      = False

    def update(self, pressed: bool, now: float) -> bool:
        """Returns True on the single tick when hold duration is first reached."""
        if pressed:
            if self._press_start is None:
                self._press_start = now
            if not self._fired and (now - self._press_start) >= self.hold_time:
                self._fired = True
                return True
        else:
            self._press_start = None
            self._fired       = False
        return False


class DoubleTapTrigger:
    """Fire on rapid double-press.

    The first press must be brief (< `max_hold`) so that a long locomotion
    hold does not accidentally arm the double-tap when released and
    immediately re-engaged.

    Timeline::

        press ──── release (≤ max_hold) ─── press (within window) ─► FIRE

    Parameters
    ----------
    window   : seconds allowed between first-release and second-press
    max_hold : maximum duration of the first press (longer → ignored)
    """

    def __init__(self, window: float = 0.6, max_hold: float = 0.35) -> None:
        self.window   = window
        self.max_hold = max_hold

        # Internal FSM:  0=idle  1=first_pressed  2=waiting_second
        self._fsm_state     = 0
        self._t_press_start = 0.0
        self._t_first_release = 0.0
        self._prev          = False

    def update(self, pressed: bool, now: float) -> bool:
        """Returns True on rising edge of second press (within window)."""
        fired   = False
        rising  = pressed and not self._prev
        falling = not pressed and self._prev

        if self._fsm_state == 0:                    # idle
            if rising:
                self._fsm_state     = 1
                self._t_press_start = now

        elif self._fsm_state == 1:                  # first press is down
            if falling:
                hold_dur = now - self._t_press_start
                if hold_dur <= self.max_hold:       # short tap → arm second
                    self._fsm_state       = 2
                    self._t_first_release = now
                else:                               # long hold → discard
                    self._fsm_state = 0

        elif self._fsm_state == 2:                  # waiting for second tap
            if now - self._t_first_release > self.window:
                self._fsm_state = 0                 # timeout
            elif rising:
                fired           = True
                self._fsm_state = 0

        self._prev = pressed
        return fired


# ── Robot feature manager ────────────────────────────────────────────────────

class RobotFeatureManager:
    """Translate finger combos + glove state into robot command modifiers.

    Call update() every control tick.  It returns a dict used by the main loop:

        cmd = feat.update(combo, raw_vx, raw_vy, raw_wz, hovering, sim_time)
        x_vel       = cmd["x_vel"]
        y_vel       = cmd["y_vel"]
        ang_z       = cmd["ang_z"]
        z_pos       = cmd["z_pos"]         # desired COM height
        euler_shift = cmd["euler_shift"]   # True → apply tilt
        hovering    = cmd["hovering"]      # True → glove in HOVERING state
    """

    STAND_Z  = 0.27   # normal standing height (m)
    CROUCH_Z = 0.17   # crouched height (m)

    def __init__(self, ctrl_dt: float = 0.005) -> None:
        self._z_pos       = self.STAND_Z
        self._is_crouched = False
        self._status      = "IDLE"

        # Posture toggle: Ring alone (0x02) — one tap = flip stand/crouch
        self._posture_trig = ToggleTrigger()

        # RecoveryStand: Middle+Ring (0x03) double-tap
        # max_hold=0.35s prevents accidental trigger when going in/out of loco
        self._recovery_trig = DoubleTapTrigger(window=0.6, max_hold=0.35)

        # Hover stand-up/down oscillation
        self._hover_entry_time: float | None = None

    # ── Main entry point ─────────────────────────────────────────────────────

    def update(
        self,
        combo:    int,
        raw_vx:   float,
        raw_vy:   float,
        raw_wz:   float,
        hovering: bool,
        t:        float,
    ) -> dict:
        """Process finger combo + glove hovering state → command dict.

        Parameters
        ----------
        combo    : fingerCombo bitmask from GloveController
        raw_vx/vy/wz : velocity from GloveController (zero when not ON)
        hovering : True when glove is in HOVERING state (magnet at mid-distance)
        t        : simulation time (seconds)
        """

        # ── 1. Posture toggle (Ring alone, 0x02) ─────────────────────────
        if self._posture_trig.update(combo == COMBO_RING):
            self._is_crouched = self._posture_trig.state
            self._z_pos = self.CROUCH_Z if self._is_crouched else self.STAND_Z

        # ── 2. RecoveryStand (Middle+Ring double-tap, 0x03) ──────────────
        if self._recovery_trig.update(combo == COMBO_MIDDLE_RING, t):
            self._is_crouched = False
            self._z_pos       = self.STAND_Z
            self._status      = "RECOVERY"
            return self._cmd(0.0, 0.0, 0.0, euler_shift=False, hovering=False)

        # ── 3. HOVERING state: glove at intermediate distance ─────────────
        if hovering:
            if self._hover_entry_time is None:
                self._hover_entry_time = t
            hover_elapsed = t - self._hover_entry_time
            if hover_elapsed >= 2.0:
                phase = (hover_elapsed - 2.0) % 1.0
                self._z_pos = self.CROUCH_Z if phase >= 0.5 else self.STAND_Z
            self._status = "HOVER"
            return self._cmd(0.0, 0.0, 0.0, euler_shift=False, hovering=True)
        elif self._hover_entry_time is not None:
            # Exiting hover — restore z_pos to the user-set posture
            self._hover_entry_time = None
            self._z_pos = self.CROUCH_Z if self._is_crouched else self.STAND_Z

        # ── 4. Locomotion combos (M+R or M+R+P) ──────────────────────────
        if combo in _LOCO_COMBOS:
            euler_shift = (combo == COMBO_MIDDLE_RING_PINKY)
            self._status = "LOCO+EULER" if euler_shift else "LOCO"
            return self._cmd(raw_vx, raw_vy, raw_wz, euler_shift=euler_shift,
                             hovering=False)

        # ── 5. Everything else → stand still ─────────────────────────────
        _LABELS = {
            COMBO_NONE:      "IDLE",
            COMBO_MIDDLE:    "MID[VR]",    # other team
            COMBO_RING:      "RING[PST]",
            COMBO_RING_PINKY:"R+P[ARM]",   # other team
        }
        self._status = _LABELS.get(combo, f"COMBO:{combo:03b}")
        return self._cmd(0.0, 0.0, 0.0, euler_shift=False, hovering=False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _cmd(self, vx: float, vy: float, wz: float,
             euler_shift: bool, hovering: bool) -> dict:
        return {
            "x_vel":       vx,
            "y_vel":       vy,
            "ang_z":       wz,
            "z_pos":       self._z_pos,
            "euler_shift": euler_shift,
            "hovering":    hovering,
        }

    def status_str(self) -> str:
        """One-line status for terminal display."""
        posture = "CROUCH" if self._is_crouched else "STAND "
        return f"{self._status:<12s} {posture} z={self._z_pos:.2f}"
