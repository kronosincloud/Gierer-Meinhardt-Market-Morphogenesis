"""
═══════════════════════════════════════════════════════════════
MODULE 3: VISUAL — Static 1920x1080 PNG Rendering
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
import config as cfg


def log(msg):
    from datetime import datetime
    print(f"[VISUAL] {datetime.now().strftime('%H:%M:%S')} | {msg}")


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


def render_static(results, data_bundle):
    """Render the full static dashboard PNG."""
    log("Rendering static PNG...")

    T = cfg.THEME
    FONT = T["FONT"]
    N = cfg.N_NODES
    N_t = results["N_t"]
    A = results["activator_surface"]
    kappa = results["kappa_trace"]
    kappa_c = results["kappa_critical"]
    turing = results["turing_flag"]
    morph_list = results["morphology"]
    D_a = results["D_a_trace"]
    D_h = results["D_h_trace"]

    # ── Figure setup — ALL backgrounds explicitly black ──────
    fig = plt.figure(
        figsize=cfg.CONFIG["FIG_SIZE"],
        dpi=cfg.CONFIG["DPI"],
        facecolor=T["BG"],
    )
    fig.patch.set_facecolor(T["BG"])

    gs = GridSpec(
        4, 2, width_ratios=[2.2, 1],
        left=0.05, right=0.97, top=0.87, bottom=0.07,
        hspace=0.38, wspace=0.10,
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_kappa = fig.add_subplot(gs[0, 1])
    ax_morph = fig.add_subplot(gs[1, 1])
    ax_heat = fig.add_subplot(gs[2, 1])
    ax_diff = fig.add_subplot(gs[3, 1])

    # ══════════════════════════════════════════════════════════
    # LEFT: 3D ACTIVATOR SURFACE
    # ══════════════════════════════════════════════════════════

    # CRITICAL: set 3D axes facecolor to black — without this,
    # matplotlib renders a white border around the 3D box
    ax3d.set_facecolor(T["BG"])

    # Dark panes
    pane = (0.02, 0.02, 0.02, 1.0)
    ax3d.xaxis.set_pane_color(pane)
    ax3d.yaxis.set_pane_color(pane)
    ax3d.zaxis.set_pane_color(pane)

    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis._axinfo["grid"]["color"] = (0.12, 0.12, 0.12, 0.6)
        axis._axinfo["grid"]["linewidth"] = 0.4

    X_mesh, Y_mesh = np.meshgrid(np.arange(N), np.arange(N_t), indexing="ij")
    Z = A.copy()

    z_max = Z.max()
    z_min = Z.min()
    norm = Normalize(vmin=z_min, vmax=z_max)

    ax3d.plot_surface(
        X_mesh, Y_mesh, Z,
        cmap=cfg.CMAP_MORPHO, norm=norm,
        alpha=0.92, rstride=1, cstride=1,
        edgecolor=(1.0, 0.08, 0.58, 0.08),
        linewidth=0.20, antialiased=True, zorder=2,
    )

    z_floor = z_min - 0.15 * (z_max - z_min + 1e-6)
    ax3d.contourf(
        X_mesh, Y_mesh, Z,
        zdir="z", offset=z_floor,
        cmap=cfg.CMAP_MORPHO, alpha=0.28, levels=16,
    )

    # Turing onset markers
    for ti in range(1, N_t):
        if turing[ti] == 1 and turing[ti - 1] == 0:
            ax3d.plot(
                [0, N - 1], [ti, ti],
                [z_floor, z_max * 1.08],
                color=T["ORANGE"], alpha=0.25, lw=1.5, ls="--",
                zorder=12,
            )

    for boundary in [5, 10, 15, 20, 25]:
        ax3d.plot(
            [boundary, boundary], [0, N_t - 1],
            [z_floor, z_floor],
            color=T["SPINE"], alpha=0.35, lw=0.6, zorder=3,
        )

    for si, (sector, col) in enumerate(cfg.SECTOR_COLORS.items()):
        x_pos = si * 5 + 2
        ax3d.text(
            x_pos, N_t - 1, z_max * 1.10,
            sector[:4], fontsize=7, color=col,
            ha="center", va="bottom",
            fontfamily=FONT, fontweight="bold", zorder=20,
        )

    mean_a = A.mean(axis=0)
    ax3d.plot(
        np.full(N_t, N / 2), np.arange(N_t), mean_a,
        color=T["YELLOW"], lw=2.0, alpha=0.8, zorder=15,
    )
    ax3d.scatter(
        [N / 2], [N_t - 1], [mean_a[-1]],
        s=28, color=T["YELLOW"],
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
    ax3d.view_init(elev=28, azim=-60)

    # ══════════════════════════════════════════════════════════
    # RIGHT PANEL 1: TURING INSTABILITY PARAMETER
    # ══════════════════════════════════════════════════════════
    _style_ax(ax_kappa)
    t_arr = np.arange(N_t)

    ax_kappa.fill_between(
        t_arr, kappa, kappa_c,
        where=kappa < kappa_c,
        color=T["GREEN"], alpha=0.12, interpolate=True,
    )
    ax_kappa.fill_between(
        t_arr, kappa, kappa_c,
        where=kappa >= kappa_c,
        color=T["MAGENTA"], alpha=0.15, interpolate=True,
    )

    ax_kappa.plot(t_arr, kappa, color=T["CYAN"], lw=1.2, label=r"$\kappa$")
    ax_kappa.plot(
        t_arr, kappa_c, color=T["ORANGE"],
        lw=1.0, ls="--", label=r"$\kappa_c$",
    )

    unstable_mask = kappa >= kappa_c
    if unstable_mask.any():
        mid = np.argmax(unstable_mask)
        ax_kappa.text(
            mid, kappa[mid] * 1.08, "TURING\nINSTABILITY",
            fontsize=6, color=T["MAGENTA"], ha="center",
            fontfamily=FONT, fontweight="bold", alpha=0.8,
        )
    stable_mask = ~unstable_mask
    if stable_mask.any():
        mid_s = np.argmax(stable_mask)
        ax_kappa.text(
            mid_s, kappa[mid_s] * 0.80, "STABLE",
            fontsize=6, color=T["GREEN"], ha="center",
            fontfamily=FONT, fontweight="bold", alpha=0.7,
        )

    ax_kappa.set_title(
        r"TURING INSTABILITY PARAMETER  $\kappa = D_h/D_a$",
        fontsize=8, color=T["TEXT_DIM"], fontfamily=FONT, pad=6,
    )
    ax_kappa.set_ylabel(r"$\kappa$", fontsize=9,
                        color=T["TEXT_DIM"], fontfamily=FONT)
    leg = ax_kappa.legend(loc="upper left", fontsize=6,
                          facecolor=T["BG"], edgecolor=T["GRID"])
    for txt in leg.get_texts():
        txt.set_color(T["TEXT_DIM"])

    # ══════════════════════════════════════════════════════════
    # RIGHT PANEL 2: PATTERN MORPHOLOGY TIMELINE
    # HOMOGENEOUS color uses visible dark blue-grey with edge
    # outline so it's always distinguishable from PANEL_BG
    # ══════════════════════════════════════════════════════════
    _style_ax(ax_morph)

    morph_colors = {
        "HOMOGENEOUS": "#1a1a2e",
        "MONOPOLE":    T["YELLOW"],
        "STRIPE":      T["BLUE"],
        "SPOT":        T["ORANGE"],
        "LABYRINTH":   T["MAGENTA"],
    }
    morph_edge_colors = {
        "HOMOGENEOUS": "#333333",
        "MONOPOLE":    T["YELLOW"],
        "STRIPE":      T["BLUE"],
        "SPOT":        T["ORANGE"],
        "LABYRINTH":   T["MAGENTA"],
    }
    morph_order = ["HOMOGENEOUS", "MONOPOLE", "STRIPE", "SPOT", "LABYRINTH"]

    morph_matrix = np.zeros((len(morph_order), N_t))
    for ti_m, m in enumerate(morph_list):
        if m in morph_order:
            morph_matrix[morph_order.index(m), ti_m] = 1.0

    bottom = np.zeros(N_t)
    for mi, mname in enumerate(morph_order):
        ax_morph.fill_between(
            t_arr, bottom, bottom + morph_matrix[mi],
            color=morph_colors[mname], alpha=0.8,
            label=mname.capitalize(),
            edgecolor=morph_edge_colors[mname],
            linewidth=0.3,
        )
        bottom += morph_matrix[mi]

    ax_morph.set_xlim(0, N_t - 1)
    ax_morph.set_ylim(0, 1)
    ax_morph.set_yticks([])
    ax_morph.set_title(
        "PATTERN MORPHOLOGY  (diffusion-driven instability)",
        fontsize=8, color=T["TEXT_DIM"], fontfamily=FONT, pad=6,
    )
    leg2 = ax_morph.legend(loc="upper right", fontsize=5, ncol=2,
                           facecolor=T["BG"], edgecolor=T["GRID"])
    for txt in leg2.get_texts():
        txt.set_color(T["TEXT_DIM"])

    # ══════════════════════════════════════════════════════════
    # RIGHT PANEL 3: ACTIVATOR FIELD HEATMAP
    # ══════════════════════════════════════════════════════════
    _style_ax(ax_heat)

    ax_heat.imshow(
        A, aspect="auto", origin="lower",
        cmap=cfg.CMAP_MORPHO, norm=norm,
        extent=[0, N_t - 1, -0.5, N - 0.5],
        interpolation="bilinear",
    )

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

    # ══════════════════════════════════════════════════════════
    # RIGHT PANEL 4: DIFFUSION RATES
    # ══════════════════════════════════════════════════════════
    _style_ax(ax_diff)

    ax_diff.plot(t_arr, D_a, color=T["ORANGE"], lw=1.0,
                 label=r"$D_a$ (momentum)")
    ax_diff.set_ylabel(r"$D_a$", fontsize=9, color=T["ORANGE"],
                       fontfamily=FONT)
    ax_diff.tick_params(axis="y", colors=T["ORANGE"])

    ax_diff2 = ax_diff.twinx()
    ax_diff2.set_facecolor("none")
    ax_diff2.plot(t_arr, D_h, color=T["CYAN"], lw=1.0,
                  label=r"$D_h$ (capital)")
    ax_diff2.set_ylabel(r"$D_h$", fontsize=9, color=T["CYAN"],
                        fontfamily=FONT)
    ax_diff2.tick_params(axis="y", colors=T["CYAN"], labelsize=7)
    ax_diff2.spines["right"].set_color(T["CYAN"])
    ax_diff2.spines["left"].set_color(T["ORANGE"])
    for sp_name in ["top", "bottom"]:
        ax_diff2.spines[sp_name].set_color(T["SPINE"])

    kappa_norm = kappa / (kappa.max() + 1e-10)
    ax_diff.bar(
        t_arr, -0.05 * kappa_norm, width=1.0,
        color=T["YELLOW"], alpha=0.35, zorder=0,
    )

    ax_diff.set_title(
        r"DIFFUSION RATES  $D_a$ (momentum) vs $D_h$ (capital flow)",
        fontsize=8, color=T["TEXT_DIM"], fontfamily=FONT, pad=6,
    )
    ax_diff.set_xlabel("Time window", fontsize=7,
                       color=T["TEXT_DIM"], fontfamily=FONT)

    lines1, labels1 = ax_diff.get_legend_handles_labels()
    lines2, labels2 = ax_diff2.get_legend_handles_labels()
    leg3 = ax_diff.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper left", fontsize=6,
        facecolor=T["BG"], edgecolor=T["GRID"],
    )
    for txt in leg3.get_texts():
        txt.set_color(T["TEXT_DIM"])

    # ══════════════════════════════════════════════════════════
    # TITLE BLOCK (SKILL.md Section 6)
    # ══════════════════════════════════════════════════════════

    kappa_max = kappa.max()
    n_turing = int(turing.sum())
    non_homog = [m for m in morph_list if m != "HOMOGENEOUS"]
    if non_homog:
        dom_morph = Counter(non_homog).most_common(1)[0][0]
    else:
        dom_morph = "HOMOGENEOUS"

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

    out_path = cfg.CONFIG["STATIC_PNG"]
    fig.savefig(out_path, dpi=cfg.CONFIG["DPI"], facecolor=T["BG"])
    plt.close(fig)
    log(f"Saved: {out_path}")