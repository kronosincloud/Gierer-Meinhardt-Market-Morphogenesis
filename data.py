"""
═══════════════════════════════════════════════════════════════
MODULE 1: DATA — Synthetic Market Data Generation
Gierer-Meinhardt Market Morphogenesis
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import config as cfg


def log(msg):
    print(f"[DATA] {datetime.now().strftime('%H:%M:%S')} | {msg}")


def _build_base_correlation():
    """Build sector-clustered correlation matrix."""
    N = cfg.N_NODES
    C = np.full((N, N), 0.25)
    for sec, idxs in cfg.SECTOR_IDX.items():
        for i in idxs:
            for j in idxs:
                if i != j:
                    C[i, j] = 0.65
    np.fill_diagonal(C, 1.0)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 0.01)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C


def fetch_all():
    """Generate full synthetic dataset."""
    log("Generating synthetic market data...")

    T = cfg.CONFIG["T_DAYS"]
    N = cfg.N_NODES
    seed = cfg.CONFIG["SEED"]
    rng = np.random.RandomState(seed)

    C = _build_base_correlation()
    L = np.linalg.cholesky(C)

    base_vols = rng.uniform(0.18, 0.38, N)

    # GARCH(1,1) — innovations generated independently to prevent
    # the positive-feedback overflow that occurred when using
    # already-scaled returns as GARCH input
    omega_g = 5e-6
    alpha_g = 0.06
    beta_g  = 0.92
    sigma2_0 = omega_g / (1.0 - alpha_g - beta_g)
    sigma2 = np.full(T, sigma2_0)
    garch_shocks = rng.randn(T)

    for t in range(1, T):
        sigma2[t] = omega_g + alpha_g * garch_shocks[t - 1]**2 + beta_g * sigma2[t - 1]
        sigma2[t] = np.clip(sigma2[t], sigma2_0 * 0.1, sigma2_0 * 100.0)

    garch_scale = np.sqrt(sigma2 / sigma2_0)
    garch_scale = np.clip(garch_scale, 0.3, 6.0)

    Z = rng.randn(T, N)
    X = Z @ L.T
    daily_vols = base_vols / np.sqrt(252.0)
    log_rets = X * daily_vols[np.newaxis, :] * garch_scale[:, np.newaxis]

    # ── Stress Regime 1: Mild correction (days 80-110) ──────
    for t in range(80, 110):
        progress = (t - 80) / 30.0
        bell = np.sin(np.pi * progress)
        log_rets[t] *= (1.0 + 1.2 * bell)

    # ── Stress Regime 2: SECTOR ROTATION (days 230-280) ─────
    for t in range(230, 280):
        progress = (t - 230) / 50.0
        bell = np.sin(np.pi * progress)
        log_rets[t] *= (1.0 + 0.8 * bell)
        for i, ticker in enumerate(cfg.TICKERS):
            sector = cfg.TICKER_SECTOR[ticker]
            if sector in ["TECHNOLOGY", "CONSUMER"]:
                log_rets[t, i] += 0.012 * bell
            elif sector in ["ENERGY", "INDUSTRIALS"]:
                log_rets[t, i] -= 0.008 * bell

    # ── Stress Regime 3: Crash (days 380-420) ───────────────
    for t in range(380, 420):
        progress = (t - 380) / 40.0
        bell = np.sin(np.pi * progress)
        log_rets[t] *= (1.0 + 3.5 * bell)
        log_rets[t] -= 0.003 * bell

    # ── Compute vol_proxy ───────────────────────────────────
    vol_w = cfg.CONFIG["VOL_W"]
    vol_proxy = np.zeros(T)

    # Compute all at once, then fill initial values
    for t in range(vol_w, T):
        window_rets = log_rets[t - vol_w:t]
        realised = np.sqrt(np.mean(window_rets**2, axis=0) * 252.0)
        vol_proxy[t] = np.clip(np.mean(realised), 0.05, 2.0)

    # Forward-fill initial values from first valid
    first_valid = vol_proxy[vol_w]
    vol_proxy[:vol_w] = first_valid

    # Safety: replace any remaining NaN/Inf
    vol_proxy = np.nan_to_num(vol_proxy, nan=first_valid, posinf=2.0, neginf=0.05)

    start_date = datetime(2023, 1, 3)
    dates = pd.DatetimeIndex([
        start_date + timedelta(days=int(d * 365 / 252))
        for d in range(T)
    ])

    stress = [
        (80, 110, "mild_correction"),
        (230, 280, "sector_rotation"),
        (380, 420, "crash"),
    ]

    returns_df = pd.DataFrame(log_rets, index=dates, columns=cfg.TICKERS)

    log(f"Generated {T} days x {N} stocks")
    log(f"Vol proxy range: [{vol_proxy.min():.3f}, {vol_proxy.max():.3f}]")
    log(f"Vol proxy at rotation peak (~day 255): {vol_proxy[255]:.3f}")
    log(f"Vol proxy at crash peak (~day 400): {vol_proxy[400]:.3f}")
    log(f"Any NaN in log_rets: {np.any(np.isnan(log_rets))}")

    return {
        "returns":     returns_df,
        "dates":       dates,
        "vol_proxy":   vol_proxy,
        "tickers":     cfg.TICKERS,
        "stress":      stress,
        "log_rets":    log_rets,
    }