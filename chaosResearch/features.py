"""Chaos metric feature engineering: Lyapunov, Hurst, Permutation Entropy.
Optimized for performance."""
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from math import factorial
from scipy.spatial import cKDTree


# ---- Permutation Entropy (Bandt-Pompe) ----
def perm_entropy(x, m=5, tau=1, normalize=True):
    """Compute permutation entropy of time series x."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    needed = (m - 1) * tau + 1
    if n < needed:
        return np.nan
    fact_m = factorial(m)
    n_patterns = n - (m - 1) * tau
    # Build embedded matrix efficiently
    indices = np.arange(m) * tau
    embedded = np.array([x[i + indices] for i in range(n_patterns)])
    # Convert to ordinal patterns using a hash
    ranks = np.argsort(np.argsort(embedded, axis=1), axis=1)
    # Encode ranks as single integers
    multipliers = m ** np.arange(m)
    codes = (ranks * multipliers).sum(axis=1)
    _, counts = np.unique(codes, return_counts=True)
    probs = counts / counts.sum()
    H = -np.sum(probs * np.log(probs))
    if normalize:
        H /= np.log(fact_m)
    return H


# ---- Hurst Exponent via DFA ----
def hurst_dfa(x, min_window=4, max_window=None, n_windows=15, order=1):
    """Detrended Fluctuation Analysis for Hurst exponent."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if max_window is None:
        max_window = n // 4
    if max_window <= min_window or n < 16:
        return np.nan
    windows = np.unique(
        np.floor(np.logspace(np.log10(min_window),
                             np.log10(max_window), n_windows)).astype(int))
    windows = windows[windows >= 4]
    Y = np.cumsum(x - np.mean(x))
    F_list = []
    valid_windows = []
    for s in windows:
        Ns = n // s
        if Ns < 2:
            continue
        # Vectorized segment processing
        segments = Y[:Ns * s].reshape(Ns, s)
        t = np.arange(s, dtype=np.float64)
        rms_vals = np.zeros(Ns)
        for v in range(Ns):
            coeffs = np.polyfit(t, segments[v], order)
            fit = np.polyval(coeffs, t)
            rms_vals[v] = np.sqrt(np.mean((segments[v] - fit) ** 2))
        F_list.append(np.mean(rms_vals))
        valid_windows.append(s)
    if len(F_list) < 3:
        return np.nan
    logs = np.log(np.array(valid_windows, dtype=np.float64))
    logF = np.log(np.array(F_list, dtype=np.float64))
    mask = np.isfinite(logs) & np.isfinite(logF)
    if mask.sum() < 3:
        return np.nan
    H = np.polyfit(logs[mask], logF[mask], 1)[0]
    return H


# ---- Rosenstein Maximal Lyapunov Exponent ----
def lyap_rosenstein(x, emb_dim=3, tau=1, theiler=None, k_max=40, fs=1.0):
    """Estimate maximal Lyapunov exponent via Rosenstein's method."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if theiler is None:
        theiler = emb_dim * tau
    M = N - (emb_dim - 1) * tau
    if M <= 2 * emb_dim + 10:
        return np.nan, None, None
    # Build embedding matrix efficiently
    emb = np.column_stack([x[d * tau: d * tau + M] for d in range(emb_dim)])
    tree = cKDTree(emb)
    # Query nearest neighbors (enough to find one beyond Theiler window)
    k_query = min(theiler + 5, M - 1)
    if k_query < 2:
        return np.nan, None, None
    dists, inds = tree.query(emb, k=k_query)
    neighbors = -np.ones(M, dtype=int)
    for i in range(M):
        for ki in range(1, k_query):
            j = inds[i, ki]
            if abs(i - j) > theiler:
                neighbors[i] = j
                break
    valid_idx = np.where(neighbors >= 0)[0]
    if len(valid_idx) < 10:
        return np.nan, None, None
    k_max = min(k_max, M // 5)
    if k_max < 5:
        return np.nan, None, None
    # Compute divergence curves vectorized
    ln_d = np.full((len(valid_idx), k_max), np.nan)
    for k in range(k_max):
        i_shifted = valid_idx + k
        j_shifted = neighbors[valid_idx] + k
        valid_k = (i_shifted < M) & (j_shifted < M)
        if valid_k.any():
            diffs = emb[i_shifted[valid_k]] - emb[j_shifted[valid_k]]
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))
            ln_d[valid_k, k] = np.log(np.maximum(distances, 1e-12))
    mean_ln = np.nanmean(ln_d, axis=0)
    ks = np.arange(k_max, dtype=np.float64)
    # Fit slope on initial linear region
    k_lin = max(5, k_max // 4)
    valid_fit = np.isfinite(mean_ln[:k_lin])
    if valid_fit.sum() < 3:
        return np.nan, None, None
    slope, intercept, r_value, _, _ = linregress(
        ks[:k_lin][valid_fit] / fs, mean_ln[:k_lin][valid_fit])
    return slope, r_value ** 2, mean_ln


# ---- Rolling Feature Computation ----
def compute_features_df(prices_df, w_short=30, w_med=125, w_long=250,
                        emb_dim=3, tau=1, perm_m=5, verbose=True):
    """Compute all rolling features for the prices DataFrame.
    Using w_long=250 (1 year) for faster computation while still being robust."""
    df = prices_df.copy()
    df['vol_short'] = df['ret'].rolling(w_short).std()
    arr_ret = df['ret'].values
    n = len(df)
    hurst_vals = np.full(n, np.nan)
    perm_vals = np.full(n, np.nan)
    lyap_vals = np.full(n, np.nan)
    theiler = emb_dim * tau
    # Compute every 1 day but with step for speed
    for t in range(n):
        if t >= w_med - 1:
            window = arr_ret[t - w_med + 1: t + 1]
            hurst_vals[t] = hurst_dfa(window, n_windows=12)
            perm_vals[t] = perm_entropy(window, m=perm_m, tau=1)
        if t >= w_long - 1:
            window = arr_ret[t - w_long + 1: t + 1]
            window_d = detrend(window, type='linear')
            lam, _, _ = lyap_rosenstein(
                window_d, emb_dim=emb_dim, tau=tau,
                theiler=theiler, k_max=40)
            lyap_vals[t] = lam
        if verbose and t > 0 and t % 1000 == 0:
            print(f"  Features: {t}/{n} days done")
    df['hurst'] = hurst_vals
    df['perm_entropy'] = perm_vals
    df['lyap'] = lyap_vals
    if verbose:
        print(f"  Features: {n}/{n} days done")
    return df
