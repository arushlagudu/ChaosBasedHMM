"""Surrogate testing for chaos metric validation (IAAFT)."""
import numpy as np
from features import lyap_rosenstein, perm_entropy


def iaaft_surrogate(x, max_iter=100, tol=1e-6):
    """Generate one IAAFT (Iterated Amplitude Adjusted Fourier Transform) surrogate.
    Preserves both the amplitude distribution and power spectrum of x.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    sorted_x = np.sort(x)
    amp_orig = np.abs(np.fft.rfft(x))
    # Initialize with random shuffle
    surr = x.copy()
    np.random.shuffle(surr)
    for _ in range(max_iter):
        # Step 1: impose spectral amplitudes
        phase = np.angle(np.fft.rfft(surr))
        surr_fft = amp_orig * np.exp(1j * phase)
        surr_new = np.fft.irfft(surr_fft, n=n)
        # Step 2: rank-order to match original amplitude distribution
        rank = np.argsort(np.argsort(surr_new))
        surr_prev = surr.copy()
        surr = sorted_x[rank]
        # Convergence check
        if np.max(np.abs(surr - surr_prev)) < tol:
            break
    return surr


def surrogate_test_lyapunov(x, n_surrogates=99, emb_dim=3, tau=1,
                            theiler=None, k_max=50, alpha=0.05):
    """Test if Lyapunov exponent is significantly different from surrogates."""
    from scipy.signal import detrend
    x_d = detrend(x, type='linear')
    obs_lyap, _, _ = lyap_rosenstein(x_d, emb_dim=emb_dim, tau=tau,
                                     theiler=theiler, k_max=k_max)
    if np.isnan(obs_lyap):
        return obs_lyap, np.nan, np.nan, False
    surr_lyaps = []
    for i in range(n_surrogates):
        s = iaaft_surrogate(x)
        s_d = detrend(s, type='linear')
        sl, _, _ = lyap_rosenstein(s_d, emb_dim=emb_dim, tau=tau,
                                   theiler=theiler, k_max=k_max)
        if not np.isnan(sl):
            surr_lyaps.append(sl)
    if len(surr_lyaps) < 10:
        return obs_lyap, np.nan, np.nan, False
    surr_lyaps = np.array(surr_lyaps)
    p_value = np.mean(surr_lyaps >= obs_lyap)
    surr_mean = np.mean(surr_lyaps)
    significant = (p_value < alpha)
    return obs_lyap, surr_mean, p_value, significant


def surrogate_test_perm_entropy(x, n_surrogates=99, m=5, tau=1, alpha=0.05):
    """Test if permutation entropy is significantly different from surrogates."""
    obs_pe = perm_entropy(x, m=m, tau=tau)
    if np.isnan(obs_pe):
        return obs_pe, np.nan, np.nan, False
    surr_pes = []
    for i in range(n_surrogates):
        s = iaaft_surrogate(x)
        spe = perm_entropy(s, m=m, tau=tau)
        if not np.isnan(spe):
            surr_pes.append(spe)
    if len(surr_pes) < 10:
        return obs_pe, np.nan, np.nan, False
    surr_pes = np.array(surr_pes)
    # For PE, we expect observed < surrogate if deterministic structure exists
    p_value = np.mean(surr_pes <= obs_pe)
    surr_mean = np.mean(surr_pes)
    significant = (p_value < alpha)
    return obs_pe, surr_mean, p_value, significant
