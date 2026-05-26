"""
═══════════════════════════════════════════════════════════════
MODULE 4: ANIMATE — 120-Frame GIF Rendering
Gierer-Meinhardt Market Morphogenesis
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from collections import Counter
import imageio
import config as cfg


def log(msg):
    from datetime import datetime
    print(f"[ANIMATE] {datetime.now().strftime('%H:%M:%S')} | {msg}")


def _ease_quintic(x):
    """Quintic ease-in-out for smooth transitions."""
    x = np.clip(x, 0.0, 1.0)
    return 6*x**5 - 15*x**4 + 10*x**3


def _style_ax(ax):
    """Apply standard 2D panel styling per SKILL.md."""
    ax.set_facecolor(cfg.THEME["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(cfg.THEME["SPINE"])
        sp.set_linewidth(0.5)
    ax.tick_params(
        colors=cfg.THEME["TEXT_DIM"], labelsize=7,
        direction="in", length=3,
    )
    ax.yaxis.grid(True, color=cfg.THEME["GRID"], lw=0.3, alpha=0.4)
    ax.xaxis.grid(True, color=cfg.THEME["GRID"], lw=0.3, alpha=0.25)


def _canvas_to_rgb(fig):
    """Extract RGB pixel array from figure canvas."""
    fig.canvas.draw()
    try:
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[:, :, :3]
    except AttributeError:
        rgb_str = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(rgb_str, dtype=np.uint8).reshape(h, w, 3)


def _build_schedule(N_t, N_grow=45, N_hold=20, N_orbit=55):
    """
    Build per-frame camera and data schedule for 3-phase animation.

    ORBIT: Pure continuous 360° rotation. No pauses, no slowdowns.
    55 frames linearly spaced from progress 0.0 to 1.0.
    """
    schedule = []

    # ── Phase 1: GROW ───────────────────────────────────────
    for i in range(N_grow):
        raw = i / max(1, N_grow - 1)
        eased = _ease_quintic(raw)
        tc = max(2, int(eased * N_t))
        z_scale = 0.05 + 0.95 * eased
        elev = 5.0 + 23.0 * eased
        azim = -60.0 + 5.0 * eased
        schedule.append({
            "phase": "GROW", "t_cutoff": tc, "z_scale": z_scale,
            "elev": elev, "azim": azim, "progress": raw,
        })

    # ── Phase 2: HOLD ───────────────────────────────────────
    for i in range(N_hold):
        breath = np.sin(2 * np.pi * i / N_hold)
        schedule.append({
            "phase": "HOLD", "t_cutoff": N_t, "z_scale": 1.0,
            "elev": 28.0 + 2.0 * breath,
            "azim": -55.0 + 5.0 * (i / N_hold),
            "progress": i / N_hold,
        })

    # ── Exact end state of HOLD (orbit starts from here) ─────
    hold_end_i = N_hold - 1
    hold_end_azim = -55.0 + 5.0 * (hold_end_i / N_hold)
    hold_end_elev = 28.0 + 2.0 * np.sin(2 * np.pi * hold_end_i / N_hold)

    # ── Phase 3: ORBIT — continuous, no stops ───────────────
    orbit_progs = np.linspace(0.0, 1.0, N_orbit)

    for orb_prog in orbit_progs:
        azim = hold_end_azim + 360.0 * orb_prog
        elev = hold_end_elev + 20.0 * np.sin(np.pi * orb_prog * 1.4)
        schedule.append({
            "phase": "ORBIT", "t_cutoff": N_t, "z_scale": 1.0,
            "elev": elev, "azim": azim, "progress": orb_prog,
        })

    return schedule


def render_animation(results, data_bundle):
    """Render 120-frame GIF matching the static PNG exactly."""
    log("Building animation schedule...")

    T = cfg.THEME
    FONT = T["FONT"]
    N = cfg.N_NODES
    N_t = results["N_t"]
    A = results["activator_surface"]
    kappa = results["kappa_trace"]
    kappa_c = results["kappa_critical"]
    turing = results["turing_flag"]
    morph_list = results["morphology"]
    dominant_k = results["dominant_k"]
    D_a = results["D_a_trace"]
    D_h = results["D_h_trace"]

    N_grow = cfg.CONFIG["N_GROW"]
    N_hold = cfg.CONFIG["N_HOLD"]
    N_orbit = cfg.CONFIG["N_ORBIT"]
    total_frames = N_grow + N_hold + N_orbit

    schedule = _build_schedule(N_t, N_grow, N_hold, N_orbit)

    z_max = A.max()
    z_min = A.min()
    norm = Normalize(vmin=z_min, vmax=z_max)

    X_full, Y_full = np.meshgrid(np.arange(N), np.arange(N_t), indexing="ij")

    morph_colors = {
        "HOMOGENEOUS": "#1a1a2e", "MONOPOLE": T["YELLOW"],
        "STRIPE": T["BLUE"], "SPOT": T["ORANGE"], "LABYRINTH": T["MAGENTA"],
    }
    morph_edge_colors = {
        "HOMOGENEOUS": "#333333", "MONOPOLE": T["YELLOW"],
        "STRIPE": T["BLUE"], "SPOT": T["ORANGE"], "LABYRINTH": T["MAGENTA"],
    }
    morph_order = ["HOMOGENEOUS", "MONOPOLE", "STRIPE", "SPOT", "LABYRINTH"]
    morph_matrix = np.zeros((len(morph_order), N_t))
    for ti_m, m in enumerate(morph_list):
        if m in morph_order:
            morph_matrix[morph_order.index(m), ti_m] = 1.0

    non_homog = [m for m in morph_list if m != "HOMOGENEOUS"]
    dom_morph = Counter(non_homog).most_common(1)[0][0] if non_homog else "HOMOGENEOUS"
    kappa_max = kappa.max()
    n_turing = int(turing.sum())

    phase_colors = {"GROW": T["CYAN"], "HOLD": T["GREEN"], "ORBIT": T["PURPLE"]}

    frames = []
    log(f"Rendering {total_frames} frames...")

    for fi, sched in enumerate(schedule):
        if fi % 20 == 0:
            log(f"  Frame {fi+1}/{total_frames}  [{sched['phase']}]")

        tc = sched["t_cutoff"]
        z_scale = sched["z_scale"]
        elev = sched["elev"]
        azim = sched["azim"]
        phase = sched["phase"]

        fig = plt.figure(
            figsize=cfg.CONFIG["FIG_SIZE"],
            dpi=cfg.CONFIG["GIF_DPI"],
            facecolor=T["BG"],
        )
        fig.patch.set_facecolor(T["BG"])

        gs = GridSpec(
            4, 2, width_ratios=[2.2, 1],
            left=0.05, right=0.97, top=0.87, bottom=0.07,
            hspace=0.38, wspace=0.10, figure=fig,
        )

        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax_kappa = fig.add_subplot(gs[0, 1])
        ax_morph = fig.add_subplot(gs[1, 1])
        ax_heat = fig.add_subplot(gs[2, 1])
        ax_diff = fig.add_subplot(gs[3, 1])

        ax3d.set_facecolor(T["BG"])
        _style_ax(ax_kappa)
        _style_ax(ax_morph)
        _style_ax(ax_heat)
        _style_ax(ax_diff)

        # ── 3D SURFACE ──────────────────────────────────────
        pane = (0.02, 0.02, 0.02, 1.0)
        ax3d.xaxis.set_pane_color(pane)
        ax3d.yaxis.set_pane_color(pane)
        ax3d.zaxis.set_pane_color(pane)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis._axinfo["grid"]["color"] = (0.12, 0.12, 0.12, 0.6)
            axis._axinfo["grid"]["linewidth"] = 0.4

        A_vis = A[:, :tc].copy() * z_scale
        X_vis, Y_vis = X_full[:, :tc], Y_full[:, :tc]
        nt_vis = A_vis.shape[1]

        if nt_vis > 1 and z_scale > 0.02:
            ax3d.plot_surface(
                X_vis, Y_vis, A_vis,
                cmap=cfg.CMAP_MORPHO, norm=norm,
                alpha=0.92, rstride=1, cstride=1,
                edgecolor=(1.0, 0.08, 0.58, 0.08),
                linewidth=0.20, antialiased=True, zorder=2,
            )

            z_fl = (z_min * z_scale) - 0.15 * (z_max * z_scale - z_min * z_scale + 1e-6)
            ax3d.contourf(
                X_vis, Y_vis, A_vis, zdir="z", offset=z_fl,
                cmap=cfg.CMAP_MORPHO, alpha=0.28, levels=16,
            )

            for ti_mark in range(1, tc):
                if turing[ti_mark] == 1 and turing[ti_mark - 1] == 0:
                    ax3d.plot(
                        [0, N-1], [ti_mark, ti_mark], [z_fl, z_max * z_scale * 1.08],
                        color=T["ORANGE"], alpha=0.25, lw=1.5, ls="--", zorder=12,
                    )

            for boundary in [5, 10, 15, 20, 25]:
                ax3d.plot(
                    [boundary, boundary], [0, nt_vis - 1], [z_fl, z_fl],
                    color=T["SPINE"], alpha=0.35, lw=0.6, zorder=3,
                )

            for si, (sector, col) in enumerate(cfg.SECTOR_COLORS.items()):
                x_pos = si * 5 + 2
                ax3d.text(
                    x_pos, nt_vis - 1, z_max * z_scale * 1.10,
                    sector[:4], fontsize=7, color=col, ha="center", va="bottom",
                    fontfamily=FONT, fontweight="bold", zorder=20,
                )

            mean_a = A_vis.mean(axis=0)
            ax3d.plot(
                np.full(nt_vis, N/2), np.arange(nt_vis), mean_a,
                color=T["YELLOW"], lw=2.0, alpha=0.8, zorder=15,
            )
            ax3d.scatter(
                [N/2], [nt_vis - 1], [mean_a[-1]], s=28, color=T["YELLOW"],
                edgecolor="white", linewidth=0.5, zorder=16,
            )

        ax3d.set_xlabel("ASSET NODE  i", fontsize=10, fontweight="bold",
                        color=T["TEXT_DIM"], labelpad=12, fontfamily=FONT)
        ax3d.set_ylabel("TIME WINDOW  t", fontsize=10, fontweight="bold",
                        color=T["TEXT_DIM"], labelpad=12, fontfamily=FONT)
        ax3d.set_zlabel(r"$a_i(t)$", fontsize=13, fontweight="bold",
                        color=T["TEXT_DIM"], labelpad=10, fontfamily=FONT)
        ax3d.tick_params(axis="both", colors=T["TEXT_DIM"], labelsize=7)
        ax3d.set_box_aspect([2.0, 1.8, 0.85])
        ax3d.view_init(elev=elev, azim=azim)

        # ── PANEL 1: KAPPA ──────────────────────────────────
        t_arr_vis = np.arange(tc)
        ax_kappa.fill_between(
            t_arr_vis, kappa[:tc], kappa_c[:tc],
            where=kappa[:tc] < kappa_c[:tc],
            color=T["GREEN"], alpha=0.12, interpolate=True,
        )
        ax_kappa.fill_between(
            t_arr_vis, kappa[:tc], kappa_c[:tc],
            where=kappa[:tc] >= kappa_c[:tc],
            color=T["MAGENTA"], alpha=0.15, interpolate=True,
        )
        ax_kappa.plot(t_arr_vis, kappa[:tc], color=T["CYAN"], lw=1.2)
        ax_kappa.plot(t_arr_vis, kappa_c[:tc], color=T["ORANGE"], lw=1.0, ls="--")
        if tc > 0:
            ax_kappa.scatter(
                [tc-1], [kappa[tc-1]], s=20, color=T["YELLOW"],
                edgecolor="white", linewidth=0.5, zorder=10,
            )
        ax_kappa.set_xlim(0, N_t - 1)
        ax_kappa.set_title(
            r"TURING INSTABILITY PARAMETER  $\kappa = D_h/D_a$",
            fontsize=8, color=T["TEXT_DIM"], fontfamily=FONT, pad=6,
        )
        ax_kappa.set_ylabel(r"$\kappa$", fontsize=9,
                            color=T["TEXT_DIM"], fontfamily=FONT)

        # ── PANEL 2: MORPHOLOGY ─────────────────────────────
        bottom = np.zeros(tc)
        for mi, mname in enumerate(morph_order):
            ax_morph.fill_between(
                t_arr_vis, bottom, bottom + morph_matrix[mi, :tc],
                color=morph_colors[mname], alpha=0.8,
                edgecolor=morph_edge_colors[mname], linewidth=0.3,
            )
            bottom += morph_matrix[mi, :tc]
        ax_morph.set_xlim(0, N_t - 1)
        ax_morph.set_ylim(0, 1)
        ax_morph.set_yticks([])
        ax_morph.set_title(
            "PATTERN MORPHOLOGY  (diffusion-driven instability)",
            fontsize=8, color=T["TEXT_DIM"], fontfamily=FONT, pad=6,
        )

        # ── PANEL 3: HEATMAP ────────────────────────────────
        ax_heat.imshow(
            A, aspect="auto", origin="lower",
            cmap=cfg.CMAP_MORPHO, norm=norm,
            extent=[0, N_t - 1, -0.5, N - 0.5],
            interpolation="bilinear",
        )
        if tc > 0:
            ax_heat.axvline(tc - 1, color=T["YELLOW"], lw=1.5, alpha=0.9)
        for boundary in [5, 10, 15, 20, 25]:
            ax_heat.axhline(boundary - 0.5, color=T["SPINE"], lw=0.5, alpha=0.5)
        ax_heat.set_yticks([2, 7, 12, 17, 22, 27])
        ax_heat.set_yticklabels(
            ["TECH", "FIN", "HEALTH", "ENERGY", "CONS", "IND"],
            fontsize=6, color=T["TEXT_DIM"], fontfamily=FONT,
        )
        ax_heat.set_title(
            r"ACTIVATOR FIELD  $a(i,t)$  [momentum patterns]",
            fontsize=8, color=T["TEXT_DIM"], fontfamily=FONT, pad=6,
        )
        ax_heat.set_xlabel("Time window", fontsize=7,
                           color=T["TEXT_DIM"], fontfamily=FONT)

        # ── PANEL 4: DIFFUSION RATES ────────────────────────
        ax_diff.plot(t_arr_vis, D_a[:tc], color=T["ORANGE"], lw=1.0)
        ax_diff.set_ylabel(r"$D_a$", fontsize=9, color=T["ORANGE"],
                           fontfamily=FONT)
        ax_diff.tick_params(axis="y", colors=T["ORANGE"])
        ax_diff2 = ax_diff.twinx()
        ax_diff2.set_facecolor("none")
        ax_diff2.plot(t_arr_vis, D_h[:tc], color=T["CYAN"], lw=1.0)
        ax_diff2.set_ylabel(r"$D_h$", fontsize=9, color=T["CYAN"],
                            fontfamily=FONT)
        ax_diff2.tick_params(axis="y", colors=T["CYAN"], labelsize=7)
        ax_diff2.spines["right"].set_color(T["CYAN"])
        ax_diff2.spines["left"].set_color(T["ORANGE"])
        for sp_name in ["top", "bottom"]:
            ax_diff2.spines[sp_name].set_color(T["SPINE"])
        kappa_norm = kappa[:tc] / (kappa[:tc].max() + 1e-10)
        ax_diff.bar(
            t_arr_vis, -0.05 * kappa_norm, width=1.0,
            color=T["YELLOW"], alpha=0.35, zorder=0,
        )
        ax_diff.set_xlim(0, N_t - 1)
        ax_diff.set_title(
            r"DIFFUSION RATES  $D_a$ (momentum) vs $D_h$ (capital flow)",
            fontsize=8, color=T["TEXT_DIM"], fontfamily=FONT, pad=6,
        )
        ax_diff.set_xlabel("Time window", fontsize=7,
                           color=T["TEXT_DIM"], fontfamily=FONT)

        # ── TITLE BLOCK ─────────────────────────────────────
        fig.text(
            0.50, 0.965,
            "GIERER-MEINHARDT MORPHOGENESIS  |  SECTOR ROTATION AS TURING PATTERN",
            ha="center", fontsize=22, fontweight="bold",
            color=T["ORANGE"], fontfamily=FONT,
        )
        rho_val = cfg.PARAMS["rho"]
        mu_a_val = cfg.PARAMS["mu_a"]
        mu_h_val = cfg.PARAMS["mu_h"]
        fig.text(
            0.50, 0.937,
            f"$\\partial_t a_i = D_a \\tilde{{L}}_{{ij}} a_j "
            f"+ \\rho\\, a_i^2/h_i - \\mu_a\\, a_i + \\rho_a$"
            f"$\\qquad \\partial_t h_i = D_h \\tilde{{L}}_{{ij}} h_j "
            f"+ \\rho\\, a_i^2 - \\mu_h\\, h_i$"
            f"     "
            f"$\\rho={rho_val}$   $\\mu_a={mu_a_val}$   $\\mu_h={mu_h_val}$",
            ha="center", fontsize=11,
            color=T["TEXT_DIM"], fontfamily=FONT,
        )
        fig.text(
            0.97, 0.907,
            f"\u03ba = D_h/D_a  |  peak = {kappa_max:.2f}  "
            f"|  Turing windows = {n_turing}/{N_t}  "
            f"|  Dominant morphology: {dom_morph}",
            ha="right", fontsize=10, fontweight="bold",
            color=T["YELLOW"], fontfamily=FONT,
        )
        fig.text(
            0.985, 0.010, T["WATERMARK"],
            ha="right", va="bottom", fontsize=10,
            color=T["TEXT_DIM"], fontfamily=FONT, alpha=0.6,
        )

        # ── HUD OVERLAYS ────────────────────────────────────
        ax3d.text2D(
            0.02, 0.97, f"\u25c6 {phase}",
            transform=ax3d.transAxes, fontsize=11,
            color=phase_colors[phase], fontfamily=FONT, fontweight="bold",
        )
        if tc > 0:
            ti_now = min(tc - 1, N_t - 1)
            k_now = kappa[ti_now]
            kc_now = kappa_c[ti_now]
            morph_now = morph_list[ti_now]
            dk_now = dominant_k[ti_now]
            status = "TURING UNSTABLE" if k_now >= kc_now else "STABLE"
            status_col = T["MAGENTA"] if k_now >= kc_now else T["GREEN"]
            ax3d.text2D(
                0.02, 0.91,
                f"\u03ba = {k_now:.2f}  |  \u03ba_c = {kc_now:.2f}  |  {status}",
                transform=ax3d.transAxes, fontsize=8,
                color=status_col, fontfamily=FONT,
            )
            ax3d.text2D(
                0.02, 0.86,
                f"Pattern: {morph_now}  |  k = {dk_now}",
                transform=ax3d.transAxes, fontsize=8,
                color=T["TEXT_DIM"], fontfamily=FONT,
            )

        prog = fi / max(1, total_frames - 1)
        bar_filled = int(prog * 30)
        bar_empty = 30 - bar_filled
        ax3d.text2D(
            0.02, 0.02,
            "\u2593" * bar_filled + "\u2591" * bar_empty,
            transform=ax3d.transAxes, fontsize=7,
            color=T["ORANGE"], fontfamily=FONT,
        )

        frame = _canvas_to_rgb(fig)
        frames.append(frame)
        plt.close(fig)

    out_path = cfg.CONFIG["ANIM_GIF"]
    log(f"Encoding {len(frames)} frames to GIF...")
    imageio.mimsave(out_path, frames, fps=cfg.CONFIG["GIF_FPS"], loop=0)
    log(f"Saved: {out_path}")