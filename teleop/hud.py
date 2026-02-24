"""Glove teleoperation HUD — live matplotlib instrument panel.

Runs in a background daemon thread; main loop writes to a shared
TeleopState dataclass and the HUD redraws at ~20 Hz.

Usage (in run_teleop.py)::

    from teleop.hud import TeleopState, TeleopHUD
    state = TeleopState()
    hud   = TeleopHUD(state)
    hud.start()          # opens matplotlib window in background thread

    # Inside the control loop:
    state.update(sim_time=..., glove_state=..., combo=...,
                 vx=..., vy=..., wz=..., status=..., z_pos=...)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Shared state (written by main loop, read by HUD thread)
# ---------------------------------------------------------------------------

@dataclass
class TeleopState:
    sim_time:    float = 0.0
    glove_state: str   = "OFF"
    combo:       int   = 0x00
    vx:          float = 0.0
    vy:          float = 0.0
    wz:          float = 0.0
    status:      str   = "IDLE"
    z_pos:       float = 0.27
    # history for the G-force strip charts (last N samples)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _hist_vx: list = field(default_factory=list, repr=False)
    _hist_vy: list = field(default_factory=list, repr=False)
    _hist_wz: list = field(default_factory=list, repr=False)
    _hist_t:  list = field(default_factory=list, repr=False)
    HIST_LEN: int  = 200   # samples kept (~10 s at 20 Hz)

    def update(
        self,
        sim_time: float,
        glove_state: str,
        combo: int,
        vx: float,
        vy: float,
        wz: float,
        status: str,
        z_pos: float,
    ) -> None:
        with self._lock:
            self.sim_time    = sim_time
            self.glove_state = glove_state
            self.combo       = combo
            self.vx          = vx
            self.vy          = vy
            self.wz          = wz
            self.status      = status
            self.z_pos       = z_pos
            self._hist_t.append(sim_time)
            self._hist_vx.append(vx)
            self._hist_vy.append(vy)
            self._hist_wz.append(wz)
            if len(self._hist_t) > self.HIST_LEN:
                self._hist_t  = self._hist_t[-self.HIST_LEN:]
                self._hist_vx = self._hist_vx[-self.HIST_LEN:]
                self._hist_vy = self._hist_vy[-self.HIST_LEN:]
                self._hist_wz = self._hist_wz[-self.HIST_LEN:]

    def snapshot(self) -> dict:
        with self._lock:
            return dict(
                sim_time    = self.sim_time,
                glove_state = self.glove_state,
                combo       = self.combo,
                vx          = self.vx,
                vy          = self.vy,
                wz          = self.wz,
                status      = self.status,
                z_pos       = self.z_pos,
                hist_t      = list(self._hist_t),
                hist_vx     = list(self._hist_vx),
                hist_vy     = list(self._hist_vy),
                hist_wz     = list(self._hist_wz),
            )


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

_COMBO_LABEL = {
    0x00: "---",
    0x01: "M--",
    0x02: "-R-",
    0x03: "MR-",
    0x04: "--P",
    0x06: "-RP",
    0x07: "MRP",
}

_STATE_COLOR = {
    "OFF":     "#555555",
    "WAITING": "#FFA500",
    "ON":      "#00CC44",
    "DAMP":    "#6699FF",
}


class TeleopHUD:
    """Live instrument panel rendered by matplotlib in a daemon thread.

    Layout
    ------
    ┌──────────────────────────────────────┐
    │  G-force strip chart  (vx, vy, wz)  │  top half
    ├──────────────────────────────────────┤
    │  Polar velocity radar                │  bottom-left
    │  Digital readout + state badge       │  bottom-right
    └──────────────────────────────────────┘
    """

    def __init__(self, state: TeleopState, hz: float = 20.0) -> None:
        self._state   = state
        self._dt      = 1.0 / hz
        self._thread  = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ── background thread ────────────────────────────────────────────────

    def _run(self) -> None:
        import matplotlib
        matplotlib.use("TkAgg")          # non-interactive backend that works in threads
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np

        plt.ion()
        fig = plt.figure("Go2 Teleop HUD", figsize=(5.4, 5))
        fig.patch.set_facecolor("#1a1a2e")

        gs = gridspec.GridSpec(
            2, 2,
            figure=fig,
            left=0.07, right=0.97,
            top=0.93,  bottom=0.10,
            hspace=0.45, wspace=0.35,
        )

        # ── strip chart (top, spans both columns) ────────────────────────
        ax_strip = fig.add_subplot(gs[0, :])
        ax_strip.set_facecolor("#0d0d1a")
        ax_strip.set_title("Velocity history", color="white", fontsize=9)
        ax_strip.set_ylabel("m/s  |  rad/s", color="#aaaaaa", fontsize=8)
        ax_strip.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax_strip.spines.values():
            spine.set_edgecolor("#333355")
        ax_strip.set_xlim(0, self._state.HIST_LEN)
        ax_strip.set_ylim(-2.0, 2.0)
        ax_strip.axhline(0, color="#333355", linewidth=0.8)

        (line_vx,) = ax_strip.plot([], [], color="#FF4444", lw=1.2, label="vx")
        (line_vy,) = ax_strip.plot([], [], color="#44FF88", lw=1.2, label="vy")
        (line_wz,) = ax_strip.plot([], [], color="#4488FF", lw=1.2, label="wz")
        ax_strip.legend(
            loc="upper left", fontsize=7,
            facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white",
        )

        # ── polar radar (bottom-left) ─────────────────────────────────────
        ax_polar = fig.add_subplot(gs[1, 0], projection="polar")
        ax_polar.set_facecolor("#0d0d1a")
        ax_polar.set_title("Velocity vector", color="white", fontsize=9, pad=8)
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.tick_params(colors="#555577", labelsize=7)
        ax_polar.set_ylim(0, 1.0)
        ax_polar.set_rlabel_position(45)
        ax_polar.yaxis.set_tick_params(labelsize=6, colors="#555577")
        for spine in ax_polar.spines.values():
            spine.set_edgecolor("#333355")

        (polar_arrow,) = ax_polar.plot([], [], "o-", color="#FF4444", lw=2, ms=5)
        circle_max = ax_polar.plot(
            np.linspace(0, 2 * np.pi, 200), [1.0] * 200,
            "--", color="#333355", lw=0.8,
        )
        # wz arc indicator
        (wz_arc,) = ax_polar.plot([], [], color="#4488FF", lw=2, alpha=0.7)

        # ── digital readout (bottom-right) ───────────────────────────────
        ax_dig = fig.add_subplot(gs[1, 1])
        ax_dig.set_facecolor("#0d0d1a")
        ax_dig.axis("off")

        txt_time   = ax_dig.text(0.05, 0.92, "t=0.0 s",         color="#aaaaaa", fontsize=8,  transform=ax_dig.transAxes)
        txt_state  = ax_dig.text(0.05, 0.75, "GLOVE: OFF",       color="#555555", fontsize=11, transform=ax_dig.transAxes, fontweight="bold")
        txt_combo  = ax_dig.text(0.05, 0.58, "Fingers: ---",     color="#aaaaaa", fontsize=9,  transform=ax_dig.transAxes)
        txt_vx     = ax_dig.text(0.05, 0.44, "vx:  +0.00 m/s",  color="#FF4444", fontsize=9,  transform=ax_dig.transAxes, family="monospace")
        txt_vy     = ax_dig.text(0.05, 0.32, "vy:  +0.00 m/s",  color="#44FF88", fontsize=9,  transform=ax_dig.transAxes, family="monospace")
        txt_wz     = ax_dig.text(0.05, 0.20, "wz:  +0.00 r/s",  color="#4488FF", fontsize=9,  transform=ax_dig.transAxes, family="monospace")
        txt_status = ax_dig.text(0.05, 0.06, "IDLE   z=0.27",   color="#888888", fontsize=8,  transform=ax_dig.transAxes)

        plt.show(block=False)

        max_lin_vel = 0.8   # for normalising polar plot

        while self._running:
            t0  = time.perf_counter()
            s   = self._state.snapshot()
            n   = len(s["hist_t"])

            # ── strip chart ───────────────────────────────────────────────
            xs  = list(range(n))
            line_vx.set_data(xs, s["hist_vx"])
            line_vy.set_data(xs, s["hist_vy"])
            line_wz.set_data(xs, s["hist_wz"])
            ax_strip.set_xlim(0, max(self._state.HIST_LEN, n))

            # ── polar arrow ───────────────────────────────────────────────
            import numpy as np
            speed  = np.hypot(s["vx"], s["vy"])
            r_norm = min(speed / max(max_lin_vel, 1e-3), 1.0)
            angle  = np.arctan2(s["vy"], s["vx"])   # atan2(y, x) → standard math angle
            # theta_zero=N, direction=CW: forward(vx>0) → North(theta=0)
            theta  = -angle
            polar_arrow.set_data([theta, theta], [0, r_norm])

            # wz arc: show rotation as a small arc
            if abs(s["wz"]) > 0.05:
                arc_half = min(abs(s["wz"]) / 1.5 * np.pi, np.pi)
                arc_sign = np.sign(s["wz"])
                arc_angles = np.linspace(-arc_half * arc_sign,
                                          arc_half * arc_sign, 40)
                wz_arc.set_data(arc_angles, [0.85] * 40)
            else:
                wz_arc.set_data([], [])

            # ── digital readout ───────────────────────────────────────────
            clr = _STATE_COLOR.get(s["glove_state"], "#888888")
            combo_lbl = _COMBO_LABEL.get(s["combo"], f"{s['combo']:03b}")

            txt_time.set_text(f"t = {s['sim_time']:6.1f} s")
            txt_state.set_text(f"GLOVE: {s['glove_state']}")
            txt_state.set_color(clr)
            txt_combo.set_text(f"Fingers: [{combo_lbl}]")
            txt_vx.set_text(f"vx:  {s['vx']:+.2f} m/s")
            txt_vy.set_text(f"vy:  {s['vy']:+.2f} m/s")
            txt_wz.set_text(f"wz:  {s['wz']:+.2f} r/s")
            txt_status.set_text(f"{s['status']:<12s}  z={s['z_pos']:.2f}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # pace to target Hz
            elapsed = time.perf_counter() - t0
            sleep   = self._dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        plt.close(fig)
