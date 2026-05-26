"""
═══════════════════════════════════════════════════════════════
MODULE 2: ENGINE — Gierer-Meinhardt Reaction-Diffusion Solver
Gierer-Meinhardt Market Morphogenesis
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.integrate import odeint
import config as cfg


def log(msg):
    from datetime import datetime
    print(f"[ENGINE] {datetime.now().strftime('%H:%M:%S')} | {msg}")


def build_graph_laplacian(corr_matrix):
    """
    Build normalised graph Laplacian from rolling correlation matrix.

    Biology: Tissue connectivity — how fast chemicals diffuse between
    neighbouring cells in the embryonic epithelium.

    Finance: Market information-flow network — how quickly momentum
    signals and capital availability propagate between correlated assets.

    Math:
        W_ij = max(0, corr_ij)
        L_sym = D^{-1/2}(D - W)D^{-1/2}

    Returns: L (NxN), eigenvalues (N,), eigenvectors (NxN)
    """
    N = corr_matrix.shape[0]
    W = np.maximum(corr_matrix.copy(), 0.0)
    np.fill_diagonal(W, 0.0)
    degrees = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
    D = np.diag(degrees)
    L = D_inv_sqrt @ (D - W) @ D_inv_sqrt
    L = (L + L.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return L, eigenvalues, eigenvectors


def compute_steady_state(params):
    """
    Compute homogeneous steady state (a0, h0).

    Math: a0 = (mu_h + rho_a) / mu_a,  h0 = rho * a0^2 / mu_h
    Expected: a0 ~ 1.51, h0 ~ 16.9 with rho=4.0
    """
    a0 = (params["mu_h"] + params["rho_a"]) / params["mu_a"]
    h0 = params["rho"] * a0**2 / params["mu_h"]
    return a0, h0


def compute_jacobian(a0, h0, params):
    """
    Compute 2x2 Jacobian of GM reaction terms at steady state.

    Math:
        f_a = 2*rho*a0/h0 - mu_a
        f_h = -rho*a0^2/h0^2
        g_a = 2*rho*a0
        g_h = -mu_h
    """
    rho, mu_a, mu_h = params["rho"], params["mu_a"], params["mu_h"]
    f_a = 2.0 * rho * a0 / h0 - mu_a
    f_h = -rho * a0**2 / h0**2
    g_a = 2.0 * rho * a0
    g_h = -mu_h
    return np.array([[f_a, f_h], [g_a, g_h]])


def _gm_rhs(state, t, L, D_a, D_h, params):
    """
    RHS of the discrete Gierer-Meinhardt ODE system.

    Math:
        da_i/dt = D_a * sum_j L_ij * a_j + rho*a_i^2/h_i - mu_a*a_i + rho_a
        dh_i/dt = D_h * sum_j L_ij * h_j + rho*a_i^2       - mu_h*h_i
    """
    N = len(state) // 2
    a = np.maximum(state[:N], 1e-6)
    h = np.maximum(state[N:], 1e-6)

    rho   = params["rho"]
    mu_a  = params["mu_a"]
    mu_h  = params["mu_h"]
    rho_a = params["rho_a"]

    react_a = rho * a**2 / h - mu_a * a + rho_a
    react_h = rho * a**2       - mu_h * h
    diff_a = -D_a * (L @ a)
    diff_h = -D_h * (L @ h)

    return np.concatenate([diff_a + react_a, diff_h + react_h])


def run_gierer_meinhardt(a_init, h_init, L, D_a, D_h, params):
    """
    Solve GM ODEs on the network using scipy.integrate.odeint.

    Biology: Simulates morphogenetic pattern development. With short
    integration time (T_ode=8), the system preserves and amplifies
    initial spatial perturbations rather than fully relaxing to
    the uniform steady state.

    Finance: Given current momentum (activator) and capital availability
    (inhibitor), evolves the market field forward. Short integration
    preserves the sector rotation pattern from the initial conditions
    while the a^2/h nonlinearity amplifies high-momentum nodes.

    Returns: (a_final, h_final) each of shape (N,)
    """
    N = len(a_init)
    state0 = np.concatenate([
        np.maximum(a_init, 1e-6),
        np.maximum(h_init, 1e-6),
    ])
    T_ode = params["T_ode"]
    dt = params["dt_ode"]
    t_vals = np.arange(0, T_ode, dt)

    try:
        sol = odeint(
            _gm_rhs, state0, t_vals,
            args=(L, D_a, D_h, params),
            rtol=1e-6, atol=1e-8,
            mxstep=5000,
        )
        a_final = np.maximum(sol[-1, :N], 1e-6)
        h_final = np.maximum(sol[-1, N:], 1e-6)
    except Exception:
        a_final = np.maximum(a_init, 0.01)
        h_final = np.maximum(h_init, 0.01)

    a_final = np.clip(a_final, 0.01, 50.0)
    h_final = np.clip(h_final, 0.01, 100.0)

    return a_final, h_final


def classify_pattern(a_field):
    """
    Classify spatial morphology via DFT on the ring-arranged nodes.

    Biology: Dominant wavenumber k determines pattern type:
    k=0 uniform, k=1-2 stripes, k=3-6 spots, k>6 labyrinth.

    Finance: k tells sector rotation morphology. k=4-6 means
    isolated sector bull runs (SPOT).

    Returns: (dominant_k, amplitude, morphology)
    """
    N = len(a_field)
    if N < 4:
        return 0, 0.0, "HOMOGENEOUS"

    A_fft = np.abs(np.fft.fft(a_field))
    half = N // 2
    if half < 2:
        return 0, 0.0, "HOMOGENEOUS"

    pos_spectrum = A_fft[1:half]
    dominant_k = int(np.argmax(pos_spectrum)) + 1
    amplitude = A_fft[dominant_k] / (A_fft[0] + 1e-10)

    if amplitude < 0.05:
        morphology = "HOMOGENEOUS"
    elif dominant_k <= 1:
        morphology = "MONOPOLE"
    elif dominant_k <= 3:
        morphology = "STRIPE"
    elif dominant_k <= 6:
        morphology = "SPOT"
    else:
        morphology = "LABYRINTH"

    return dominant_k, amplitude, morphology


def compute_morphogenetic_surface(data_bundle):
    """
    Master function: compute the full morphogenetic surface.

    At each time window:
      1. Rolling correlation -> graph Laplacian + eigenvalues
      2. Estimate D_a(t) and D_h(t) from market data
      3. Check Turing instability using fixed kappa_c = 3.5
      4. Initialize GM from market Sharpe / inverse vol
      5. Integrate GM ODEs (short T_ode=8 to preserve patterns)
      6. Classify morphology via FFT
    """
    log("Computing morphogenetic surface...")

    returns = data_bundle["returns"].values
    vol_proxy = data_bundle["vol_proxy"]
    T_total = returns.shape[0]
    N = cfg.N_NODES
    params = cfg.CONFIG["PARAMS"].copy()
    corr_w = cfg.CONFIG["CORR_WINDOW"]
    mom_w = cfg.CONFIG["MOMENTUM_W"]
    vol_w = cfg.CONFIG["VOL_W"]
    N_t = cfg.CONFIG["SUBSAMPLE"]
    kappa_c_fixed = cfg.KAPPA_CRITICAL_FIXED

    time_idx = np.linspace(corr_w, T_total - 1, N_t, dtype=int)

    activator_surface = np.zeros((N, N_t))
    inhibitor_surface = np.zeros((N, N_t))
    kappa_trace = np.zeros(N_t)
    kappa_critical = np.full(N_t, kappa_c_fixed)
    turing_flag = np.zeros(N_t, dtype=int)
    dominant_k_arr = np.zeros(N_t, dtype=int)
    amplitude_arr = np.zeros(N_t)
    morphology = []
    D_a_trace = np.zeros(N_t)
    D_h_trace = np.zeros(N_t)

    vp_valid = vol_proxy[vol_proxy > 0]
    vp_median = np.median(vp_valid) if len(vp_valid) > 0 else 0.2

    a0_ss, h0_ss = compute_steady_state(params)
    log(f"Steady state: a0 = {a0_ss:.4f}, h0 = {h0_ss:.4f}")

    J_ss = compute_jacobian(a0_ss, h0_ss, params)
    tr_J = np.trace(J_ss)
    det_J = np.linalg.det(J_ss)
    log(f"Jacobian: tr(J) = {tr_J:.4f} (< 0: {tr_J < 0}), "
        f"det(J) = {det_J:.4f} (> 0: {det_J > 0})")
    log(f"Using fixed kappa_c = {kappa_c_fixed} for Turing threshold")

    for ti, t in enumerate(time_idx):
        if ti % 15 == 0:
            log(f"  Window {ti+1}/{N_t}  (day {t})")

        window_rets = returns[t - corr_w:t]
        if window_rets.shape[0] < 10:
            activator_surface[:, ti] = a0_ss
            inhibitor_surface[:, ti] = h0_ss
            morphology.append("HOMOGENEOUS")
            continue

        corr = np.corrcoef(window_rets.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        L, eigvals, eigvecs = build_graph_laplacian(corr)

        # ── Estimate D_a(t) ─────────────────────────────────
        D_a = 0.5
        mom_window = returns[t - mom_w:t]
        if mom_window.shape[0] > 6:
            sector_means = mom_window.mean(axis=1)
            if np.std(sector_means) > 1e-12:
                ac = np.corrcoef(sector_means[:-5], sector_means[5:])[0, 1]
                if not (np.isnan(ac) or np.isinf(ac)):
                    D_a = np.clip(1.0 - ac, 0.05, 2.0)

        # ── Estimate D_h(t) ─────────────────────────────────
        vp = vol_proxy[t]
        if np.isnan(vp) or vp <= 0:
            vp = vp_median
        D_h = np.clip(1.0 + 3.0 * (vp / (vp_median + 1e-10)), 1.0, 15.0)

        # ── Turing instability: fixed threshold ─────────────
        kappa_t = D_h / D_a
        is_unstable = kappa_t > kappa_c_fixed

        kappa_trace[ti] = kappa_t
        kappa_critical[ti] = kappa_c_fixed
        turing_flag[ti] = 1 if is_unstable else 0
        D_a_trace[ti] = D_a
        D_h_trace[ti] = D_h

        # ── Market-derived initial conditions ───────────────
        sharpe_window = returns[t - mom_w:t]
        if sharpe_window.shape[0] > 5:
            means = sharpe_window.mean(axis=0)
            stds = sharpe_window.std(axis=0)
            stds = np.maximum(stds, 1e-10)
            sharpe = means / stds
        else:
            sharpe = np.zeros(N)

        a_init = np.clip(np.maximum(sharpe, 0.01), 0.01, 10.0)

        vol_window = returns[t - vol_w:t]
        if vol_window.shape[0] > 5:
            rv = np.sqrt(np.mean(vol_window**2, axis=0) * 252.0)
        else:
            rv = np.ones(N) * 0.2
        rv = np.nan_to_num(rv, nan=0.2, posinf=2.0)
        h_init = np.clip(1.0 / (rv + 0.01), 0.01, 50.0)

        # ── Perturbation to break symmetry ──────────────────
        rng_pert = np.random.RandomState(t)
        perturbation = 1.0 + 0.05 * rng_pert.randn(N)
        a_init = a_init * perturbation

        # ── Run GM integration (short: preserves pattern) ───
        a_final, h_final = run_gierer_meinhardt(
            a_init, h_init, L, D_a, D_h, params
        )

        activator_surface[:, ti] = a_final
        inhibitor_surface[:, ti] = h_final

        # ── Classify pattern ────────────────────────────────
        dk, amp, morph = classify_pattern(a_final)
        dominant_k_arr[ti] = dk
        amplitude_arr[ti] = amp
        morphology.append(morph)

    n_turing = int(turing_flag.sum())
    log(f"Turing-unstable windows: {n_turing}/{N_t} "
        f"({100*n_turing/N_t:.1f}%)")
    log(f"Peak kappa = {kappa_trace.max():.2f} at window "
        f"{np.argmax(kappa_trace)}")
    morph_counts = {}
    for m in morphology:
        morph_counts[m] = morph_counts.get(m, 0) + 1
    log(f"Morphology distribution: {morph_counts}")

    return {
        "activator_surface": activator_surface,
        "inhibitor_surface": inhibitor_surface,
        "kappa_trace":       kappa_trace,
        "kappa_critical":    kappa_critical,
        "turing_flag":       turing_flag,
        "dominant_k":        dominant_k_arr,
        "amplitude":         amplitude_arr,
        "morphology":        morphology,
        "D_a_trace":         D_a_trace,
        "D_h_trace":         D_h_trace,
        "time_idx":          time_idx,
        "N_t":               N_t,
    }