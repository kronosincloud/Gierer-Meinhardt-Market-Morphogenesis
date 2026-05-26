"""
═══════════════════════════════════════════════════════════════
GIERER-MEINHARDT MARKET MORPHOGENESIS
Sector Rotation as Spontaneous Turing Pattern Formation
═══════════════════════════════════════════════════════════════
Main orchestrator: DATA -> ENGINE -> VISUAL -> ANIMATE
"""

from datetime import datetime
from collections import Counter
import numpy as np
import config as cfg
from data import fetch_all
from engine import (
    compute_morphogenetic_surface,
    compute_steady_state,
    compute_jacobian,
)
from visual import render_static
from animate import render_animation


def log(msg):
    print(f"[MAIN] {datetime.now().strftime('%H:%M:%S')} | {msg}")


def main():
    log("=" * 60)
    log("GIERER-MEINHARDT MARKET MORPHOGENESIS")
    log("Sector Rotation as Turing Pattern Formation")
    log("=" * 60)

    # ── Verify steady state ─────────────────────────────────
    a0, h0 = compute_steady_state(cfg.PARAMS)
    J = compute_jacobian(a0, h0, cfg.PARAMS)
    log(f"Steady state: a0 = {a0:.4f}, h0 = {h0:.4f}")
    log(f"Jacobian trace = {np.trace(J):.4f} (expect < 0)")
    log(f"Jacobian det   = {np.linalg.det(J):.4f} (expect > 0)")

    # ══════════════════════════════════════════════════════════
    # MODULE 1: DATA
    # ══════════════════════════════════════════════════════════
    log("═══ MODULE 1: DATA ═══")
    data = fetch_all()

    # ══════════════════════════════════════════════════════════
    # MODULE 2: ENGINE
    # ══════════════════════════════════════════════════════════
    log("═══ MODULE 2: ENGINE ═══")
    results = compute_morphogenetic_surface(data)

    # ══════════════════════════════════════════════════════════
    # MODULE 3: VISUAL
    # ══════════════════════════════════════════════════════════
    log("═══ MODULE 3: VISUAL ═══")
    render_static(results, data)

    # ══════════════════════════════════════════════════════════
    # MODULE 4: ANIMATE
    # ══════════════════════════════════════════════════════════
    log("═══ MODULE 4: ANIMATE ═══")
    render_animation(results, data)

    # ══════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════
    log("=" * 60)
    log("TURING INSTABILITY STATISTICS")
    log("=" * 60)

    kappa = results["kappa_trace"]
    kappa_c = results["kappa_critical"]
    turing = results["turing_flag"]
    morph_list = results["morphology"]
    N_t = results["N_t"]
    A = results["activator_surface"]

    n_turing = int(turing.sum())
    pct_turing = 100.0 * n_turing / N_t
    peak_kappa = kappa.max()
    peak_idx = int(np.argmax(kappa))
    peak_day = results["time_idx"][peak_idx]

    log(f"Total windows:              {N_t}")
    log(f"Turing-unstable windows:    {n_turing} ({pct_turing:.1f}%)")
    log(f"Peak kappa:                 {peak_kappa:.2f} (day {peak_day})")
    log(f"Fixed kappa_c:              {kappa_c[0]:.2f}")

    morph_counts = Counter(morph_list)
    log(f"Morphology distribution:")
    for m, c in morph_counts.most_common():
        log(f"  {m:15s}: {c:3d} ({100*c/N_t:.1f}%)")

    # Sector with highest mean activation
    sector_mean_a = {}
    for sec, idxs in cfg.SECTOR_IDX.items():
        sector_mean_a[sec] = A[idxs, :].mean()
    top_sec = max(sector_mean_a, key=sector_mean_a.get)
    low_sec = min(sector_mean_a, key=sector_mean_a.get)
    log(f"Highest-activation sector:  {top_sec} "
        f"(mean a = {sector_mean_a[top_sec]:.4f})")
    log(f"Lowest-activation sector:   {low_sec} "
        f"(mean a = {sector_mean_a[low_sec]:.4f})")

    # Check rotation window morphology
    rot_start_day = 230
    rot_end_day = 280
    rot_windows = [
        i for i, d in enumerate(results["time_idx"])
        if rot_start_day <= d <= rot_end_day
    ]
    if rot_windows:
        rot_morphs = [morph_list[i] for i in rot_windows]
        rot_counts = Counter(rot_morphs)
        log(f"\nSector rotation window (days {rot_start_day}-{rot_end_day}):")
        for m, c in rot_counts.most_common():
            log(f"  {m:15s}: {c:3d}")
        spot_pct = 100 * rot_counts.get("SPOT", 0) / len(rot_windows)
        log(f"  SPOT morphology during rotation: {spot_pct:.1f}%")

    log("=" * 60)
    log(f"COMPLETE — outputs in {cfg.CONFIG['OUT_DIR']}")
    log("=" * 60)


if __name__ == "__main__":
    main()