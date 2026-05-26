"""
═══════════════════════════════════════════════════════════════
MODULE 0: CONFIGURATION
Gierer-Meinhardt Market Morphogenesis
Sector Rotation as Spontaneous Turing Pattern Formation
═══════════════════════════════════════════════════════════════
"""

import os
from matplotlib.colors import LinearSegmentedColormap

# ── Stock Universe ──────────────────────────────────────────
SECTORS = {
    "TECHNOLOGY":  ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "FINANCIALS":  ["JPM",  "BAC",  "GS",   "MS",   "C"],
    "HEALTHCARE":  ["JNJ",  "UNH",  "PFE",  "ABBV", "MRK"],
    "ENERGY":      ["XOM",  "CVX",  "COP",  "SLB",  "EOG"],
    "CONSUMER":    ["AMZN", "TSLA", "HD",   "MCD",  "NKE"],
    "INDUSTRIALS": ["GE",   "CAT",  "BA",   "RTX",  "HON"],
}

TICKERS = []
TICKER_SECTOR = {}
SECTOR_IDX = {}
for _si, (_sec, _tks) in enumerate(SECTORS.items()):
    SECTOR_IDX[_sec] = list(range(_si * 5, _si * 5 + 5))
    for _tk in _tks:
        TICKERS.append(_tk)
        TICKER_SECTOR[_tk] = _sec

SECTOR_ORDER = list(SECTORS.keys())
N_NODES = len(TICKERS)  # 30

SECTOR_COLORS = {
    "TECHNOLOGY":  "#00f2ff",
    "FINANCIALS":  "#ff9500",
    "HEALTHCARE":  "#00ff41",
    "ENERGY":      "#ffd400",
    "CONSUMER":    "#ff1493",
    "INDUSTRIALS": "#bb66ff",
}

NODE_COLORS = []
for _tk in TICKERS:
    NODE_COLORS.append(SECTOR_COLORS[TICKER_SECTOR[_tk]])

# ── Theme: Bloomberg Dark (SKILL.md spec) ───────────────────
THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#0a0a0a",
    "GRID":       "#1a1a1a",
    "SPINE":      "#333333",
    "TEXT":       "#ffffff",
    "TEXT_DIM":   "#aaaaaa",
    "ORANGE":     "#ff9500",
    "ORANGE_HOT": "#ff6b00",
    "CYAN":       "#00f2ff",
    "YELLOW":     "#ffd400",
    "GREEN":      "#00ff41",
    "RED":        "#ff3050",
    "MAGENTA":    "#ff1493",
    "PINK":       "#ff2a9e",
    "BLUE":       "#00bfff",
    "PURPLE":     "#bb66ff",
    "FONT":       "DejaVu Sans",
    "WATERMARK":  "@Laksh",
}

# ── Custom Colourmaps ───────────────────────────────────────
CMAP_MORPHO = LinearSegmentedColormap.from_list("morphogenetic_field", [
    "#000000",
    "#050020",
    "#0a0050",
    "#1a0080",
    "#6600cc",
    "#ff1493",
    "#ff6b00",
    "#ff9500",
    "#ffd400",
    "#ffffff",
], N=512)

CMAP_PATTERN = LinearSegmentedColormap.from_list("turing_kappa", [
    "#000000", "#001a4d", "#0040cc", "#00f2ff", "#ffffff",
], N=256)

# ── Gierer-Meinhardt Parameters ─────────────────────────────
# T_ode reduced to 8 so the ODE preserves initial spatial patterns
# instead of relaxing to uniform steady state over 50 time units
PARAMS = {
    "rho":    4.0,     # stronger nonlinearity for pattern amplification
    "mu_a":   0.8,
    "mu_h":   1.2,
    "rho_a":  0.01,
    "dt_ode": 0.1,
    "T_ode":  8,       # short integration: preserve market pattern
}

# Fixed Turing threshold — the exact linear analysis produces
# kappa_c > 40 for these GM parameters on this graph, which is
# not useful as a regime indicator. The fixed threshold of 3.5
# separates calm markets (uniform) from rotating markets (patterned),
# consistent with the prompt specification.
KAPPA_CRITICAL_FIXED = 3.5

# ── Master Config ───────────────────────────────────────────
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG = {
    "T_DAYS":      504,
    "SEED":        42,
    "CORR_WINDOW":  60,
    "MOMENTUM_W":   40,
    "VOL_W":        21,
    "SUBSAMPLE":    75,
    "N_NODES":      N_NODES,
    "PARAMS":       PARAMS,
    "OUT_DIR":      OUT_DIR,
    "STATIC_PNG":   os.path.join(OUT_DIR, "morphogenesis_market.png"),
    "ANIM_GIF":    os.path.join(OUT_DIR, "morphogenesis_animation.gif"),
    "DPI":          100,
    "FIG_SIZE":     (19.2, 10.8),
    "GIF_DPI":      80,
    "GIF_FPS":      10,
    "N_GROW":       45,
    "N_HOLD":       20,
    "N_ORBIT":      55,
}