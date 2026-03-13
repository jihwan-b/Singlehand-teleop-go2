"""Poster-session demo HUD for Go2 glove teleoperation.

Runs in a background daemon thread; main loop writes to DemoState and the
HUD redraws at ~15 Hz.

Layout
------
  ┌──────────────────┬─────────────────────────┐
  │  Hand visual     │  State badge + feature  │   ← 80 % of height
  │  (fingers +      │  Velocity readout       │
  │   thumb/magnet)  │  Episode stats          │
  ├──────────────────┴─────────────────────────┤
  │  Magnetometer distance gauge               │   ← 20 % of height
  └────────────────────────────────────────────┘

Usage::

    from teleop.demo_hud import DemoState, DemoHUD

    demo_state = DemoState()
    hud = DemoHUD(demo_state)
    hud.start()

    # Inside the control loop (every CTRL tick):
    demo_state.update(
        sim_time=sim_time, glove_state=ctrl_state, combo=combo,
        finger_bent=raw['finger_bent'], mag_value=mag,
        vx=x_vel, vy=y_vel, wz=ang_z,
        episode=ep, wall_bumps=wall_contacts, dist_to_goal=dist,
    )
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Shared state (written by main loop, read by HUD thread)
# ---------------------------------------------------------------------------

@dataclass
class DemoState:
    """Thread-safe snapshot object for the demo dashboard."""
    sim_time:     float      = 0.0
    glove_state:  str        = "OFF"    # OFF / WAITING / ON / HOVERING
    combo:        int        = 0x00
    finger_bent:  List[bool] = field(default_factory=lambda: [False, False, False])
    mag_value:    float      = 0.0      # 3-D magnetometer magnitude
    vx:           float      = 0.0
    vy:           float      = 0.0
    wz:           float      = 0.0
    status:       str        = "IDLE"
    episode:      int        = 1
    wall_bumps:   int        = 0
    dist_to_goal: float      = 0.0
    _lock: threading.Lock    = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if not k.startswith('_') and hasattr(self, k):
                    setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(
                sim_time     = self.sim_time,
                glove_state  = self.glove_state,
                combo        = self.combo,
                finger_bent  = list(self.finger_bent),
                mag_value    = self.mag_value,
                vx           = self.vx,
                vy           = self.vy,
                wz           = self.wz,
                status       = self.status,
                episode      = self.episode,
                wall_bumps   = self.wall_bumps,
                dist_to_goal = self.dist_to_goal,
            )


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "OFF":     "#555555",
    "WAITING": "#FFA500",
    "ON":      "#00CC44",
    "HOVERING":"#6699FF",
}
_STATE_DESC = {
    "OFF":     "magnet far away",
    "WAITING": "detected — activating…",
    "ON":      "active — steering",
    "HOVERING":"mid-range — hovering",
}
_COMBO_FEATURE = {
    0x00: "IDLE",
    0x01: "VR  (Middle)",
    0x02: "Posture  (Ring)",
    0x03: "Locomotion  (Mid + Ring)",
    0x04: "(Pinky)",
    0x06: "Arm  (Ring + Pinky)",
    0x07: "Loco + Lean  (All 3 fingers)",
}
_COMBO_LABEL = {
    0x00: "---", 0x01: "M--", 0x02: "-R-", 0x03: "MR-",
    0x04: "--P", 0x06: "-RP", 0x07: "MRP",
}

# Magnetometer thresholds — must match GloveConfig defaults
_T_OFF   =  500.0
_T_H_MIN =  500.0
_T_H_MAX = 1300.0
_T_ON    = 1500.0
_G_MAX   = 2200.0


# ---------------------------------------------------------------------------
# DemoHUD
# ---------------------------------------------------------------------------

class DemoHUD:
    """Live instrument panel rendered in a background daemon thread."""

    def __init__(self, state: DemoState, hz: float = 15.0) -> None:
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

    # ── background thread ─────────────────────────────────────────────────

    def _run(self) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch, Rectangle
        import numpy as np

        plt.ion()
        fig = plt.figure("Go2 Glove Dashboard", figsize=(3.5, 8))
        fig.patch.set_facecolor("#0f0f1e")
        fig.suptitle("Glove status",
                     color="#aaaadd", fontsize=11, fontweight="bold", y=0.988)

        gs = gridspec.GridSpec(
            3, 1, figure=fig,
            left=0.04, right=0.96,
            top=0.96,  bottom=0.04,
            hspace=0.30,
            height_ratios=[2.2, 3.2, 1.0],
        )

        # ── [0]  INFO PANEL ───────────────────────────────────────────────
        ax_i = fig.add_subplot(gs[0])
        ax_i.set_facecolor("#0d0d1a")
        ax_i.axis('off')

        # Large state badge
        txt_state = ax_i.text(
            0.50, 0.87, "OFF",
            color='white', fontsize=14, fontweight='bold',
            ha='center', va='center', transform=ax_i.transAxes,
            bbox=dict(facecolor='#555555', edgecolor='#888888',
                      boxstyle='round,pad=0.35', lw=2),
        )
        txt_state_sub = ax_i.text(
            0.50, 0.72, "magnet far away",
            color='#888899', fontsize=7.5, ha='center', va='center',
            transform=ax_i.transAxes,
        )
        ax_i.plot([0.0, 1.0], [0.64, 0.64], color='#2a2a5a', lw=1.0,
                  transform=ax_i.transAxes)

        txt_feature = ax_i.text(
            0.05, 0.56, "IDLE",
            color='#aaaacc', fontsize=10, fontweight='bold',
            transform=ax_i.transAxes,
        )
        txt_combo_raw = ax_i.text(
            0.55, 0.56, "Fingers: [---]",
            color='#888899', fontsize=7.5, transform=ax_i.transAxes,
        )
        ax_i.plot([0.0, 1.0], [0.48, 0.48], color='#2a2a5a', lw=1.0,
                  transform=ax_i.transAxes)

        txt_vx = ax_i.text(0.05, 0.39, "vx:  +0.00 m/s",
                            color='#FF5555', fontsize=8, family='monospace',
                            transform=ax_i.transAxes)
        txt_vy = ax_i.text(0.55, 0.39, "vy:  +0.00 m/s",
                            color='#55FF99', fontsize=8, family='monospace',
                            transform=ax_i.transAxes)
        txt_wz = ax_i.text(0.05, 0.25, "wz:  +0.00 r/s",
                            color='#5599FF', fontsize=8, family='monospace',
                            transform=ax_i.transAxes)

        ax_i.plot([0.0, 1.0], [0.17, 0.17], color='#2a2a5a', lw=1.0,
                  transform=ax_i.transAxes)
        txt_ep   = ax_i.text(0.05, 0.10, "Ep  1  |  t =   0.0 s",
                              color='#888899', fontsize=7.5, transform=ax_i.transAxes)
        txt_dist = ax_i.text(0.05, 0.02, "dist: 0.00 m   bumps: 0",
                              color='#888899', fontsize=7.5, transform=ax_i.transAxes)

        # ── [1]  HAND PANEL ───────────────────────────────────────────────
        ax_h = fig.add_subplot(gs[1])
        ax_h.set_facecolor("#0d0d1a")
        ax_h.set_xlim(-0.92, 0.92)
        ax_h.set_ylim(-0.75, 1.12)
        ax_h.set_aspect('equal')
        ax_h.axis('off')
        ax_h.set_title("Hand  (sensor view)", color="#8888bb", fontsize=9, pad=3)

        # Palm
        ax_h.add_patch(FancyBboxPatch(
            (-0.56, -0.50), 1.12, 0.68,
            boxstyle="round,pad=0.09",
            facecolor='#1a1a30', edgecolor='#3a3a7a', lw=1.8, zorder=1,
        ))

        # ── Fingers ────────────────────────────────────────────────────────
        # Two overlapping FancyBboxPatch objects per active finger.
        # Visibility is toggled in the update loop to show straight vs. bent.
        FW, FH_STR, FH_BENT, FB_Y = 0.18, 0.70, 0.22, 0.20

        _FSPEC = [                     # (name, x_center, is_active)
            ('index',  -0.39, False),
            ('middle', -0.13, True),
            ('ring',    0.13, True),
            ('pinky',   0.39, True),
        ]
        _f_str  = {}   # straight (dim) patch, shown when not bent
        _f_bent = {}   # bent (lit) patch, shown when bent
        _f_lbl  = {}

        _F_LABEL = {
            'index':  'Index\n(unused)',
            'middle': 'Middle',
            'ring':   'Ring',
            'pinky':  'Pinky',
        }
        for name, xc, active in _FSPEC:
            dim_fc = '#222233' if not active else '#2e2e55'
            dim_ec = '#333344' if not active else '#4444aa'

            # Straight version (visible by default)
            ps = FancyBboxPatch(
                (xc - FW/2, FB_Y), FW, FH_STR,
                boxstyle="round,pad=0.04",
                facecolor=dim_fc, edgecolor=dim_ec,
                lw=1.5, zorder=2, visible=True,
            )
            ax_h.add_patch(ps)
            _f_str[name] = ps

            # Bent version (hidden by default) — shorter + bright teal
            pb = FancyBboxPatch(
                (xc - FW/2, FB_Y), FW, FH_BENT,
                boxstyle="round,pad=0.04",
                facecolor='#00CC77', edgecolor='#00FF99',
                lw=2.0, zorder=2, visible=False,
            )
            ax_h.add_patch(pb)
            _f_bent[name] = pb

            lbl = ax_h.text(
                xc, FB_Y - 0.12, _F_LABEL[name],
                color='#555566' if not active else '#8888bb',
                fontsize=6.5, ha='center', va='top', zorder=3,
            )
            _f_lbl[name] = lbl

        # ── Thumb (horizontal, left of palm) — colored by magnet state ────
        thumb_p = FancyBboxPatch(
            (-0.90, -0.40), 0.39, 0.32,
            boxstyle="round,pad=0.06",
            facecolor='#444444', edgecolor='#666666',
            lw=2.0, zorder=2,
        )
        ax_h.add_patch(thumb_p)
        thumb_lbl = ax_h.text(
            -0.71, -0.24, "Thumb\n(magnet)",
            color='#888888', fontsize=6.5, ha='center', va='center', zorder=3,
        )


        # ── [2]  MAGNETOMETER GAUGE ───────────────────────────────────────
        ax_g = fig.add_subplot(gs[2])
        ax_g.set_facecolor("#0d0d1a")
        ax_g.set_title("Magnetometer status",
                       color="#8888bb", fontsize=8.5, pad=3)
        ax_g.set_xlim(0, _G_MAX)
        ax_g.set_ylim(0, 1)
        ax_g.axis('off')

        GY0, GH = 0.22, 0.50   # gauge bar y-start and height in [0,1] axes space

        _GZONES = [
            (0,       _T_OFF,   '#3a1a1a', '#CC4444', 'OFF\n(far)'),
            (_T_H_MIN,_T_H_MAX, '#1a1a3a', '#4466CC', 'HOVER'),
            (_T_H_MAX,_T_ON,    '#1e1e1e', '#666666', 'DEAD\nZONE'),
            (_T_ON,   _G_MAX,   '#1a3a1a', '#44CC66', 'ON\n(close)'),
        ]
        for z_lo, z_hi, fc, ec, label in _GZONES:
            ax_g.add_patch(Rectangle(
                (z_lo, GY0), z_hi - z_lo, GH,
                facecolor=fc, edgecolor=ec, lw=1.5, zorder=1,
            ))
            ax_g.text(
                (z_lo + z_hi) / 2, GY0 + GH + 0.04, label,
                color='#aaaacc', fontsize=7.5, ha='center', va='bottom', zorder=2,
            )

        # Threshold tick marks + labels
        for val, lbl in [(0, '0'), (_T_OFF, '500'),
                          (_T_H_MAX, '1300'), (_T_ON, '1500'),
                          (_G_MAX, f'{int(_G_MAX)}')]:
            ax_g.plot([val, val], [GY0 - 0.06, GY0], '-', color='#444466', lw=1.0, zorder=2)
            ax_g.text(val, GY0 - 0.10, lbl, color='#666688', fontsize=7,
                      ha='center', va='top', zorder=2)

        # Dynamic indicator: white vertical line + triangle pointer + value label
        (mag_line,) = ax_g.plot([0, 0], [GY0, GY0 + GH],
                                  color='white', lw=3, zorder=4)
        (mag_tri,)  = ax_g.plot([0], [GY0 + GH + 0.04], 'v',
                                  color='white', ms=9, zorder=5)
        mag_lbl     = ax_g.text(0, 0.94, "0",
                                  color='white', fontsize=8.5, fontweight='bold',
                                  ha='center', va='top', zorder=5)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show(block=False)

        # ── UPDATE LOOP ───────────────────────────────────────────────────
        while self._running:
            t0 = time.perf_counter()
            s  = self._state.snapshot()
            sc = _STATE_COLORS.get(s['glove_state'], '#555555')

            # ── Fingers (Middle=0, Ring=1, Pinky=2) ───────────────────────
            fb = s['finger_bent']
            for i, name in enumerate(('middle', 'ring', 'pinky')):
                bent = fb[i]
                _f_str[name].set_visible(not bent)
                _f_bent[name].set_visible(bent)
                _f_lbl[name].set_color('#00FFAA' if bent else '#8888bb')

            # ── Thumb (magnet state colour) ────────────────────────────────
            thumb_p.set_facecolor(sc)
            thumb_p.set_edgecolor('#ffffff' if s['glove_state'] != 'OFF' else '#666666')
            thumb_lbl.set_color('#ffffff' if s['glove_state'] != 'OFF' else '#888888')

            # ── State badge ────────────────────────────────────────────────
            txt_state.set_text(s['glove_state'])
            bp = txt_state.get_bbox_patch()
            bp.set_facecolor(sc)
            bp.set_edgecolor('#ffffff' if s['glove_state'] != 'OFF' else '#888888')
            txt_state_sub.set_text(_STATE_DESC.get(s['glove_state'], ''))

            # ── Feature label ──────────────────────────────────────────────
            feature   = _COMBO_FEATURE.get(s['combo'], f"0x{s['combo']:02x}")
            combo_lbl = _COMBO_LABEL.get(s['combo'], f"{s['combo']:03b}")
            is_on     = s['glove_state'] == 'ON'
            txt_feature.set_text(feature)
            txt_feature.set_color(sc if is_on else '#666688')
            txt_combo_raw.set_text(f"Fingers: [{combo_lbl}]")

            # ── Velocity ───────────────────────────────────────────────────
            txt_vx.set_text(f"vx:  {s['vx']:+.2f} m/s")
            txt_vy.set_text(f"vy:  {s['vy']:+.2f} m/s")
            txt_wz.set_text(f"wz:  {s['wz']:+.2f} r/s")

            # ── Episode stats ──────────────────────────────────────────────
            txt_ep.set_text(f"Ep {s['episode']:2d}  |  t = {s['sim_time']:6.1f} s")
            txt_dist.set_text(
                f"dist: {s['dist_to_goal']:.2f} m   bumps: {s['wall_bumps']}"
            )

            # ── Magnetometer gauge ─────────────────────────────────────────
            mag = float(np.clip(s['mag_value'], 0.0, _G_MAX))
            mag_line.set_xdata([mag, mag])
            mag_tri.set_xdata([mag])
            mag_lbl.set_x(mag)
            mag_lbl.set_text(f"{mag:.0f}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            elapsed = time.perf_counter() - t0
            rem = self._dt - elapsed
            if rem > 0:
                time.sleep(rem)

        plt.close(fig)
