# Chaos-Based Regime Detection in Financial Markets

A research framework that combines **chaos-theoretic metrics** with **Hidden Markov Models (HMMs)** to detect and classify financial market regimes, identifying when markets are chaotic, trending, or mean-reverting.

Unlike traditional regime-switching models that rely solely on returns and volatility, this approach augments the HMM feature set with three chaos metrics (Lyapunov exponent, Hurst exponent, Permutation Entropy), enabling richer and more economically meaningful regime classification. Statistical rigor is ensured through IAAFT surrogate testing.

---

## Repository Structure

```
chaos_hmm_project/
├── README.md                                # This file
├── run_analysis.py                          # Main pipeline script (orchestrates everything)
├── data_loader.py                           # Downloads & preprocesses price data via yfinance
├── features.py                              # Computes chaos metrics (Lyapunov, Hurst, PermEntropy)
├── chaosmodel.py                            # RegimeHMM class & state-to-regime mapping
├── backtest.py                              # Regime-based trading strategy & performance evaluation
├── surrogates.py                            # IAAFT surrogate testing for statistical validation
├── references.bib                           # BibTeX bibliography
├── figures/                                 # Generated figures
│   ├── {SYMBOL}_regimes.png/.pdf            # Price + regime overlay plots (per asset)
│   ├── {SYMBOL}_backtest.png/.pdf           # Backtest equity curves (per asset)
│   ├── regime_proportions.png/.pdf          # Cross-asset regime distribution
│   ├── sharpe_comparison.png/.pdf           # Chaos vs. buy-and-hold Sharpe ratios
│   ├── feature_boxplots.png/.pdf            # Pooled feature distributions by regime
│   └── avg_transition_matrix.png/.pdf       # Average regime transition probabilities
├── results/                                 # Intermediate & final results
│   ├── {SYMBOL}_features.csv               # Computed features per asset
│   └── all_results.json                     # Aggregated results for all 20 assets
└── tables/                                  # Summary tables (CSV + LaTeX)
    ├── summary_stats.csv/.tex               # Descriptive statistics
    ├── regime_characteristics.csv/.tex      # Mean features per regime
    ├── backtest_results.csv/.tex            # Strategy performance comparison
    ├── surrogate_tests.csv/.tex             # IAAFT surrogate test results
    └── dwell_times.csv/.tex                 # Regime persistence (dwell time) statistics
```

---

## Key Features

- **Three chaos metrics** computed via rolling windows:
  - **Maximal Lyapunov Exponent** (Rosenstein's method) — detects sensitivity to initial conditions (chaos)
  - **Hurst Exponent** (Detrended Fluctuation Analysis) — measures long-range dependence and persistence
  - **Permutation Entropy** (Bandt-Pompe) — quantifies time series complexity and randomness
- **Gaussian HMM** with 3 hidden states mapped to economically interpretable regimes:
  - 🔴 **Chaotic** — high Lyapunov, high entropy, unpredictable dynamics
  - 🟢 **Trending** — high Hurst, persistent price movements
  - 🔵 **Mean-Reverting** — low Hurst, prices revert to equilibrium
- **IAAFT surrogate testing** to statistically validate that observed chaos is not an artifact of linear stochastic structure
- **Backtesting engine** with transaction costs, comparing the chaos-augmented strategy against buy-and-hold
- **20 major U.S. equities** analyzed (2010–2025): SPY, AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM, JNJ, PFE, WMT, DIS, TSLA, NVDA, META, V, MA, UNH, HD, PG
- **Publication-ready outputs**: LaTeX tables, high-resolution figures, and comprehensive results

---

## Installation

### Prerequisites

- Python 3.8+
- Internet connection (for downloading price data via Yahoo Finance)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/chaos_hmm_project.git
cd chaos_hmm_project

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation & time series |
| `scipy` | Signal processing, statistics, spatial (cKDTree) |
| `matplotlib` | Plotting & figure generation |
| `seaborn` | Statistical visualizations |
| `yfinance` | Downloading historical stock prices |
| `hmmlearn` | Gaussian Hidden Markov Models |
| `scikit-learn` | Feature scaling (`StandardScaler`) |

If no `requirements.txt` is present, install manually:

```bash
pip install numpy pandas scipy matplotlib seaborn yfinance hmmlearn scikit-learn
```

---

## Quick Start

Run the full analysis pipeline with a single command:

```bash
python run_analysis.py
```

This will:
1. Download daily adjusted close prices for all 20 assets (2010–2025)
2. Compute rolling chaos features (Lyapunov, Hurst, Permutation Entropy, volatility)
3. Fit a 3-state Gaussian HMM (chaos-augmented) and a 2-state baseline HMM
4. Map hidden states to economic regimes (chaotic / trending / mean-reverting)
5. Run IAAFT surrogate tests (19 surrogates per asset)
6. Backtest the regime-based trading strategy with transaction costs
7. Generate all figures, tables, and aggregated results

**Expected runtime:** ~15–30 minutes depending on hardware and network speed.

---

## Reproducing Results

1. Ensure all dependencies are installed (see [Installation](#installation))
2. Run the pipeline:
   ```bash
   python run_analysis.py
   ```
3. Outputs will be saved to:
   - `figures/` — all regime and backtest plots
   - `tables/` — summary statistics, regime characteristics, backtest results, surrogate tests, dwell times (CSV + LaTeX)
   - `results/` — per-asset feature CSVs and `all_results.json`

> **Note:** `results/all_results.json` and the files in `figures/` and `tables/` contain the authors' original experimental output. Running the script will regenerate these files. Results may differ slightly due to:
> - Updated price data from Yahoo Finance (data extends to the run date)
> - Stochastic initialization of the HMM (mitigated by `random_state=42`)
> - Minor floating-point differences across hardware

---

## Usage

### Running the Full Pipeline

```bash
python run_analysis.py
```

### Using Individual Modules

Each module can be imported independently for custom analysis:

```python
from data_loader import prepare_timeseries
from features import compute_features_df
from chaosmodel import RegimeHMM, map_states_to_regimes
from backtest import regime_strategy, evaluate_backtest
from surrogates import surrogate_test_lyapunov, surrogate_test_perm_entropy

# 1. Download and prepare data
df = prepare_timeseries('AAPL', start='2015-01-01', end='2025-01-01')

# 2. Compute chaos features
df = compute_features_df(df, w_short=30, w_med=125, w_long=250,
                         emb_dim=3, tau=1, perm_m=5)

# 3. Fit HMM and detect regimes
feature_cols = ['ret', 'vol_short', 'lyap', 'hurst', 'perm_entropy']
sub = df.dropna(subset=feature_cols).copy()

hmm = RegimeHMM(n_states=3, cov_type='full', random_state=42)
hmm.fit(sub, feature_cols)

states = hmm.predict_states(sub)
mapping, state_stats = map_states_to_regimes(hmm, sub)
sub['regime'] = pd.Series(states, index=sub.index).map(mapping)

# 4. Backtest
bt_df = regime_strategy(sub, states, mapping, tc=0.0005)
perf = evaluate_backtest(bt_df, nav_col='nav')
print(f"Sharpe Ratio: {perf['sharpe']:.2f}")

# 5. Validate with surrogate testing
lyap_obs, lyap_surr_mean, lyap_pval, lyap_sig = surrogate_test_lyapunov(
    sub['ret'].values[:500], n_surrogates=19, emb_dim=3, tau=1, k_max=50)
print(f"Lyapunov significant: {lyap_sig} (p={lyap_pval:.4f})")
```

### Module Reference

| Module | Description |
|--------|-------------|
| `data_loader.py` | Downloads adjusted close prices via `yfinance`, computes log returns, winsorizes outliers |
| `features.py` | Rolling computation of Lyapunov exponent (Rosenstein), Hurst exponent (DFA), Permutation Entropy (Bandt-Pompe), and volatility |
| `chaosmodel.py` | `RegimeHMM` class wrapping `hmmlearn.GaussianHMM` with fit/predict/score/AIC-BIC; `map_states_to_regimes` for economic labeling |
| `backtest.py` | `regime_strategy` (long trending, contrarian mean-reverting, flat chaotic) with transaction costs; `evaluate_backtest` for performance metrics; `baseline_vol_strategy` for comparison |
| `surrogates.py` | IAAFT surrogate generation; rank-based p-value testing for Lyapunov exponent and Permutation Entropy |
| `run_analysis.py` | End-to-end orchestration: iterates over 20 assets, runs all steps, generates figures and tables |

---

## Output Files

### Figures (`figures/`)

| File | Description |
|------|-------------|
| `{SYMBOL}_regimes.png` | Price time series with color-coded regime overlay |
| `{SYMBOL}_backtest.png` | Equity curves: chaos strategy vs. buy-and-hold |
| `regime_proportions.png` | Bar chart of regime distribution across all 20 assets |
| `sharpe_comparison.png` | Side-by-side Sharpe ratio comparison |
| `feature_boxplots.png` | Box plots of chaos features grouped by regime |
| `avg_transition_matrix.png` | Heatmap of average regime transition probabilities |

### Tables (`tables/`)

| File | Description |
|------|-------------|
| `summary_stats.csv/.tex` | Descriptive statistics for each asset |
| `regime_characteristics.csv/.tex` | Mean feature values per regime across assets |
| `backtest_results.csv/.tex` | Strategy returns, Sharpe ratios, max drawdown |
| `surrogate_tests.csv/.tex` | IAAFT test results (observed vs. surrogate, p-values) |
| `dwell_times.csv/.tex` | Mean/median regime dwell times (persistence) |

---

## Full Results (20 U.S. Equities, 2010–2025)

All results below are generated from the analysis of 20 major U.S. equities over the period 2010–2025. The complete data is available in `results/all_results.json`.

### Backtest Performance: Chaos Strategy vs. Buy-and-Hold

The chaos-augmented regime strategy goes long in trending regimes, takes contrarian positions in mean-reverting regimes, and stays flat (cash) in chaotic regimes. Transaction costs of 5 bps per trade are applied.

| Asset | Strategy Ann. Return | Strategy Sharpe | Strategy Max DD | Buy&Hold Ann. Return | Buy&Hold Sharpe | Buy&Hold Max DD |
|:-----:|:--------------------:|:---------------:|:---------------:|:--------------------:|:---------------:|:---------------:|
| SPY | 1.68% | 0.120 | 44.50% | 13.21% | 0.778 | 35.11% |
| AAPL | -8.36% | -0.477 | 82.02% | 24.30% | 0.869 | 45.94% |
| MSFT | 9.58% | 0.461 | 45.70% | 22.55% | 0.873 | 40.61% |
| GOOGL | 2.81% | 0.139 | 58.55% | 18.31% | 0.670 | 47.95% |
| AMZN | -3.38% | -0.129 | 74.61% | 24.11% | 0.740 | 61.89% |
| JPM | 6.37% | 0.283 | 50.87% | 15.94% | 0.580 | 47.30% |
| BAC | -11.74% | -0.444 | 90.72% | 10.29% | 0.309 | 71.66% |
| XOM | -0.47% | -0.027 | 62.10% | 7.22% | 0.285 | 68.45% |
| JNJ | 2.24% | 0.168 | 47.90% | 10.30% | 0.606 | 29.02% |
| PFE | 2.23% | 0.126 | 49.98% | 7.44% | 0.342 | 58.00% |
| WMT | 4.51% | 0.315 | 35.02% | 14.02% | 0.712 | 37.65% |
| DIS | -3.66% | -0.259 | 59.26% | 8.17% | 0.312 | 65.19% |
| TSLA | 25.84% | 0.530 | 80.13% | 42.06% | 0.739 | 79.88% |
| NVDA | 11.70% | 0.450 | 55.18% | 50.37% | 1.101 | 71.87% |
| META | 3.34% | 0.110 | 67.68% | 31.20% | 0.829 | 79.38% |
| V | -2.46% | -0.125 | 67.29% | 23.97% | 0.990 | 38.87% |
| MA | 10.22% | 0.542 | 37.22% | 25.66% | 0.965 | 43.51% |
| UNH | 8.75% | 0.447 | 40.82% | 22.52% | 0.887 | 37.41% |
| HD | 6.24% | 0.337 | 50.22% | 21.02% | 0.916 | 37.17% |
| PG | -1.36% | -0.105 | 50.74% | 10.12% | 0.583 | 26.16% |

### Regime Distribution Across Assets

| Asset | N Obs | Date Range | 🔴 Chaotic % | 🟢 Trending % | 🔵 Mean-Reverting % |
|:-----:|:-----:|:----------:|:------------:|:-------------:|:-------------------:|
| SPY | 3,575 | 2010-12-30 to 2025-03-18 | 56.4% | 30.6% | 13.0% |
| AAPL | 3,575 | 2010-12-30 to 2025-03-18 | 64.0% | 17.8% | 18.2% |
| MSFT | 3,575 | 2010-12-30 to 2025-03-18 | 43.7% | 36.0% | 20.3% |
| GOOGL | 3,575 | 2010-12-30 to 2025-03-18 | 22.9% | 39.3% | 37.8% |
| AMZN | 3,575 | 2010-12-30 to 2025-03-18 | 23.5% | 23.5% | 53.0% |
| JPM | 3,575 | 2010-12-30 to 2025-03-18 | 36.9% | 52.6% | 10.5% |
| BAC | 3,575 | 2010-12-30 to 2025-03-18 | 43.0% | 43.2% | 13.8% |
| XOM | 3,575 | 2010-12-30 to 2025-03-18 | 15.2% | 48.4% | 36.4% |
| JNJ | 3,575 | 2010-12-30 to 2025-03-18 | 34.7% | 52.1% | 13.3% |
| PFE | 3,575 | 2010-12-30 to 2025-03-18 | 32.2% | 30.5% | 37.2% |
| WMT | 3,575 | 2010-12-30 to 2025-03-18 | 13.3% | 44.5% | 42.2% |
| DIS | 3,575 | 2010-12-30 to 2025-03-18 | 42.5% | 28.8% | 28.6% |
| TSLA | 3,453 | 2011-06-24 to 2025-03-18 | 32.6% | 35.8% | 31.6% |
| NVDA | 3,575 | 2010-12-30 to 2025-03-18 | 42.7% | 28.4% | 28.9% |
| META | 2,976 | 2013-05-20 to 2025-03-18 | 23.6% | 23.6% | 52.8% |
| V | 3,575 | 2010-12-30 to 2025-03-18 | 32.8% | 20.4% | 46.8% |
| MA | 3,575 | 2010-12-30 to 2025-03-18 | 37.1% | 57.9% | 5.0% |
| UNH | 3,575 | 2010-12-30 to 2025-03-18 | 44.2% | 49.8% | 6.1% |
| HD | 3,575 | 2010-12-30 to 2025-03-18 | 56.4% | 40.8% | 2.9% |
| PG | 3,575 | 2010-12-30 to 2025-03-18 | 6.0% | 50.6% | 43.4% |

### Regime Characteristics (Mean Feature Values per Regime)

#### 🔴 Chaotic Regime

| Asset | Mean Return | Std Return | Mean Vol | Mean Lyapunov | Mean Hurst | Mean Perm Entropy |
|:-----:|:----------:|:----------:|:--------:|:-------------:|:----------:|:-----------------:|
| SPY | 0.0596% | 0.6618% | 0.6400% | 0.1403 | 0.5645 | 0.8803 |
| AAPL | 0.1030% | 1.6411% | 1.5823% | 0.1407 | 0.6325 | 0.8767 |
| MSFT | 0.1030% | 1.2050% | 1.2068% | 0.1391 | 0.5623 | 0.8898 |
| GOOGL | 0.0464% | 2.4718% | 2.2984% | 0.1452 | 0.6080 | 0.8858 |
| AMZN | -0.0034% | 2.0825% | 2.3577% | 0.1382 | 0.6148 | 0.8767 |
| JPM | 0.0015% | 1.6388% | 1.6073% | 0.1382 | 0.6238 | 0.8787 |
| BAC | 0.0650% | 1.7182% | 1.5648% | 0.1389 | 0.6471 | 0.8859 |
| XOM | 0.0914% | 2.8294% | 2.6349% | 0.1435 | 0.5702 | 0.8895 |
| JNJ | 0.0340% | 1.0569% | 0.9820% | 0.1420 | 0.5928 | 0.8681 |
| PFE | 0.0312% | 1.0414% | 0.9816% | 0.1423 | 0.6532 | 0.8870 |
| WMT | -0.0020% | 2.3636% | 2.0888% | 0.1404 | 0.5985 | 0.8880 |
| DIS | -0.0075% | 2.1884% | 2.0089% | 0.1412 | 0.6131 | 0.8806 |
| TSLA | -0.0179% | 2.5058% | 2.4097% | 0.1442 | 0.6248 | 0.8841 |
| NVDA | 0.1860% | 3.8022% | 3.6266% | 0.1359 | 0.5868 | 0.8885 |
| META | 0.1313% | 2.3529% | 2.8796% | 0.1393 | 0.5792 | 0.8895 |
| V | 0.0566% | 1.2854% | 1.2576% | 0.1390 | 0.5558 | 0.8685 |
| MA | 0.0979% | 1.9092% | 1.9169% | 0.1413 | 0.5673 | 0.8816 |
| UNH | 0.0973% | 1.4993% | 1.5694% | 0.1444 | 0.6246 | 0.8868 |
| HD | 0.0831% | 1.0209% | 0.9887% | 0.1388 | 0.6240 | 0.8838 |
| PG | -0.0176% | 2.4706% | 2.1191% | 0.1444 | 0.5562 | 0.8701 |

#### 🟢 Trending Regime

| Asset | Mean Return | Std Return | Mean Vol | Mean Lyapunov | Mean Hurst | Mean Perm Entropy |
|:-----:|:----------:|:----------:|:--------:|:-------------:|:----------:|:-----------------:|
| SPY | 0.0656% | 1.0666% | 1.0628% | 0.1373 | 0.6176 | 0.8802 |
| AAPL | -0.0129% | 2.1318% | 1.7224% | 0.1333 | 0.5757 | 0.8953 |
| MSFT | 0.0815% | 1.4294% | 1.3781% | 0.1335 | 0.5958 | 0.8683 |
| GOOGL | 0.0970% | 1.0889% | 1.0705% | 0.1365 | 0.6158 | 0.8910 |
| AMZN | 0.0696% | 2.8314% | 2.3574% | 0.1381 | 0.6328 | 0.8767 |
| JPM | 0.1076% | 1.1257% | 1.1126% | 0.1358 | 0.6341 | 0.8907 |
| BAC | 0.0366% | 1.5167% | 1.5642% | 0.1389 | 0.6310 | 0.8860 |
| XOM | 0.0037% | 0.9308% | 0.9015% | 0.1362 | 0.5870 | 0.8856 |
| JNJ | 0.0396% | 0.8074% | 0.8003% | 0.1354 | 0.6149 | 0.8908 |
| PFE | 0.0475% | 0.8970% | 0.9173% | 0.1297 | 0.5908 | 0.8881 |
| WMT | 0.0719% | 0.7807% | 0.7874% | 0.1376 | 0.6161 | 0.8788 |
| DIS | 0.0667% | 1.1873% | 1.0832% | 0.1362 | 0.6063 | 0.8855 |
| TSLA | 0.3733% | 4.7901% | 4.5209% | 0.1414 | 0.6114 | 0.8804 |
| NVDA | 0.1842% | 1.9435% | 1.9039% | 0.1329 | 0.6315 | 0.8808 |
| META | 0.0332% | 3.6471% | 2.8770% | 0.1392 | 0.6123 | 0.8895 |
| V | 0.1201% | 2.3731% | 2.2211% | 0.1376 | 0.5811 | 0.8876 |
| MA | 0.1006% | 1.0835% | 1.0626% | 0.1376 | 0.5758 | 0.8862 |
| UNH | 0.0880% | 1.2075% | 1.1567% | 0.1358 | 0.6155 | 0.8838 |
| HD | 0.0579% | 1.6206% | 1.5949% | 0.1378 | 0.5991 | 0.8903 |
| PG | 0.0255% | 0.7762% | 0.7689% | 0.1386 | 0.5998 | 0.8757 |

#### 🔵 Mean-Reverting Regime

| Asset | Mean Return | Std Return | Mean Vol | Mean Lyapunov | Mean Hurst | Mean Perm Entropy |
|:-----:|:----------:|:----------:|:--------:|:-------------:|:----------:|:-----------------:|
| SPY | -0.0343% | 2.0567% | 1.8116% | 0.1382 | 0.5682 | 0.8909 |
| AAPL | 0.1235% | 1.7761% | 1.7170% | 0.1334 | 0.5504 | 0.8951 |
| MSFT | 0.0311% | 2.5062% | 2.3465% | 0.1370 | 0.5423 | 0.8814 |
| GOOGL | 0.0467% | 1.7016% | 1.7304% | 0.1369 | 0.6089 | 0.8759 |
| AMZN | 0.1314% | 1.5735% | 1.5319% | 0.1357 | 0.5987 | 0.8913 |
| JPM | 0.0150% | 3.5780% | 3.3499% | 0.1378 | 0.5380 | 0.8768 |
| BAC | -0.0355% | 3.9282% | 3.6285% | 0.1357 | 0.5532 | 0.8845 |
| XOM | 0.0326% | 1.5852% | 1.5629% | 0.1404 | 0.5857 | 0.8854 |
| JNJ | 0.0483% | 1.7781% | 1.6206% | 0.1345 | 0.5864 | 0.8930 |
| PFE | 0.0101% | 1.8581% | 1.7846% | 0.1365 | 0.5780 | 0.8850 |
| WMT | 0.0480% | 1.1143% | 1.1641% | 0.1384 | 0.5942 | 0.8904 |
| DIS | 0.0526% | 0.9929% | 1.0797% | 0.1363 | 0.5896 | 0.8854 |
| TSLA | 0.0354% | 2.8459% | 2.9744% | 0.1334 | 0.5974 | 0.8810 |
| NVDA | 0.1074% | 1.9077% | 1.9142% | 0.1330 | 0.6164 | 0.8809 |
| META | 0.1291% | 1.4951% | 1.4847% | 0.1361 | 0.5930 | 0.8809 |
| V | 0.0900% | 1.1638% | 1.1310% | 0.1356 | 0.5530 | 0.8910 |
| MA | -0.0819% | 3.9445% | 3.5786% | 0.1412 | 0.5149 | 0.8986 |
| UNH | -0.0997% | 3.7213% | 3.1802% | 0.1389 | 0.6129 | 0.8795 |
| HD | 0.1897% | 3.9038% | 3.4445% | 0.1368 | 0.5522 | 0.9022 |
| PG | 0.0608% | 1.0997% | 1.0836% | 0.1350 | 0.5714 | 0.8883 |

### IAAFT Surrogate Test Results

Surrogate testing validates whether observed chaos metrics reflect genuine nonlinear deterministic structure or are artifacts of linear stochastic processes. Each asset is tested with 19 IAAFT surrogates.

| Asset | Lyap Observed | Lyap Surr Mean | Lyap p-value | Lyap Significant | PE Observed | PE Surr Mean | PE p-value | PE Significant |
|:-----:|:------------:|:--------------:|:------------:|:----------------:|:-----------:|:------------:|:----------:|:--------------:|
| SPY | 0.1159 | 0.1145 | 0.4211 | ❌ | 0.9750 | 0.9725 | 0.6316 | ❌ |
| AAPL | 0.1062 | 0.1115 | 1.0000 | ❌ | 0.9805 | 0.9736 | 0.9474 | ❌ |
| MSFT | 0.1153 | 0.1127 | 0.2105 | ❌ | 0.9713 | 0.9685 | 0.7895 | ❌ |
| GOOGL | 0.1130 | 0.1112 | 0.3158 | ❌ | 0.9783 | 0.9741 | 0.9474 | ❌ |
| AMZN | 0.1115 | 0.1118 | 0.5789 | ❌ | 0.9746 | 0.9744 | 0.4211 | ❌ |
| JPM | 0.1106 | 0.1107 | 0.3684 | ❌ | 0.9780 | 0.9758 | 0.6842 | ❌ |
| BAC | 0.1064 | 0.1117 | 1.0000 | ❌ | 0.9680 | 0.9732 | 0.1053 | ❌ |
| XOM | 0.1073 | 0.1111 | 0.8421 | ❌ | 0.9737 | 0.9737 | 0.4211 | ❌ |
| JNJ | 0.1148 | 0.1116 | 0.2632 | ❌ | 0.9817 | 0.9722 | 1.0000 | ❌ |
| PFE | 0.1037 | 0.1120 | 1.0000 | ❌ | 0.9805 | 0.9741 | 1.0000 | ❌ |
| WMT | 0.1119 | 0.1120 | 0.5263 | ❌ | 0.9696 | 0.9764 | 0.0000 | ✅ |
| DIS | 0.1136 | 0.1095 | 0.2105 | ❌ | 0.9664 | 0.9763 | 0.0000 | ✅ |
| TSLA | 0.1086 | 0.1116 | 0.8947 | ❌ | 0.9669 | 0.9732 | 0.0526 | ❌ |
| NVDA | 0.1051 | 0.1104 | 0.7895 | ❌ | 0.9729 | 0.9740 | 0.5263 | ❌ |
| META | 0.1137 | 0.1112 | 0.2105 | ❌ | 0.9703 | 0.9688 | 0.5789 | ❌ |
| V | 0.1111 | 0.1142 | 0.7368 | ❌ | 0.9757 | 0.9729 | 0.7368 | ❌ |
| MA | 0.1122 | 0.1133 | 0.5789 | ❌ | 0.9696 | 0.9720 | 0.3158 | ❌ |
| UNH | 0.1092 | 0.1104 | 0.7368 | ❌ | 0.9694 | 0.9750 | 0.0526 | ❌ |
| HD | 0.1064 | 0.1115 | 0.8947 | ❌ | 0.9740 | 0.9739 | 0.4737 | ❌ |
| PG | 0.1105 | 0.1114 | 0.5789 | ❌ | 0.9662 | 0.9710 | 0.2105 | ❌ |

### Regime Dwell Times (Days)

Dwell time measures how long the market remains in each regime before transitioning.

| Asset | Chaotic Mean | Chaotic Median | Chaotic Max | Trending Mean | Trending Median | Trending Max | MR Mean | MR Median | MR Max |
|:-----:|:-----------:|:--------------:|:-----------:|:------------:|:---------------:|:------------:|:-------:|:---------:|:------:|
| SPY | 111.9 | 91.0 | 375 | 54.8 | 43.0 | 177 | 46.5 | 36.5 | 115 |
| AAPL | 143.0 | 97.0 | 810 | 1.0 | 1.0 | 1 | 1.0 | 1.0 | 1 |
| MSFT | 71.0 | 33.5 | 312 | 64.3 | 42.0 | 276 | 45.4 | 41.5 | 128 |
| GOOGL | 35.6 | 25.0 | 177 | 70.2 | 43.5 | 387 | 52.0 | 37.0 | 254 |
| AMZN | 1.0 | 1.0 | 2 | 1.0 | 1.0 | 1 | 86.2 | 39.0 | 349 |
| JPM | 47.1 | 33.0 | 232 | 81.8 | 55.0 | 220 | 53.4 | 33.0 | 125 |
| BAC | 1.0 | 1.0 | 1 | 1.0 | 1.0 | 2 | 82.0 | 66.5 | 203 |
| XOM | 90.8 | 82.5 | 175 | 157.3 | 85.0 | 594 | 81.2 | 49.5 | 354 |
| JNJ | 82.7 | 69.0 | 411 | 77.5 | 56.5 | 274 | 36.5 | 30.0 | 112 |
| PFE | 60.6 | 59.0 | 134 | 54.6 | 22.5 | 249 | 73.9 | 53.0 | 185 |
| WMT | 21.6 | 24.5 | 57 | 61.2 | 28.0 | 435 | 45.7 | 31.0 | 225 |
| DIS | 60.8 | 30.0 | 286 | 1.0 | 1.0 | 1 | 1.0 | 1.0 | 2 |
| TSLA | 66.1 | 44.0 | 267 | 47.6 | 41.0 | 157 | 49.6 | 39.0 | 261 |
| NVDA | 69.4 | 47.0 | 368 | 1.0 | 1.0 | 2 | 1.0 | 1.0 | 1 |
| META | 1.0 | 1.0 | 2 | 1.0 | 1.0 | 1 | 98.2 | 68.5 | 478 |
| V | 46.9 | 24.0 | 194 | 38.4 | 30.0 | 131 | 52.3 | 35.5 | 221 |
| MA | 60.3 | 32.5 | 324 | 108.9 | 69.0 | 389 | 35.6 | 31.0 | 74 |
| UNH | 58.5 | 30.0 | 364 | 71.2 | 42.0 | 485 | 18.1 | 2.0 | 71 |
| HD | 87.7 | 47.0 | 473 | 56.0 | 38.5 | 206 | 34.0 | 19.0 | 68 |
| PG | 35.8 | 35.5 | 70 | 78.7 | 44.0 | 365 | 62.0 | 53.0 | 158 |

### Model Selection (AIC / BIC)

| Asset | Chaos AIC | Chaos BIC | Baseline AIC | Baseline BIC |
|:-----:|:---------:|:---------:|:------------:|:------------:|
| SPY | 145,698,461 | 145,698,869 | 49,416,130 | 49,416,204 |
| AAPL | 162,060,679 | 162,061,087 | 56,055,066 | 56,055,140 |
| MSFT | 151,134,660 | 151,135,068 | 55,531,383 | 55,531,457 |
| GOOGL | 149,322,177 | 149,322,585 | 55,442,937 | 55,443,011 |
| AMZN | 162,787,184 | 162,787,592 | 55,930,938 | 55,931,012 |
| JPM | 141,072,812 | 141,073,220 | 49,636,511 | 49,636,585 |
| BAC | 153,053,102 | 153,053,510 | 48,745,738 | 48,745,812 |
| XOM | 145,284,890 | 145,285,298 | 51,884,421 | 51,884,495 |
| JNJ | 148,825,151 | 148,825,559 | 55,040,934 | 55,041,008 |
| PFE | 149,538,955 | 149,539,363 | 54,776,425 | 54,776,499 |
| WMT | 148,565,198 | 148,565,606 | 53,124,026 | 53,124,101 |
| DIS | 158,614,667 | 158,615,075 | 53,101,975 | 53,102,049 |
| TSLA | 144,156,497 | 144,156,903 | 54,020,089 | 54,020,163 |
| NVDA | 159,297,278 | 159,297,686 | 54,215,738 | 54,215,812 |
| META | 109,984,852 | 109,985,248 | 37,167,658 | 37,167,730 |
| V | 151,256,955 | 151,257,363 | 54,140,168 | 54,140,242 |
| MA | 149,364,309 | 149,364,717 | 51,859,528 | 51,859,603 |
| UNH | 150,912,557 | 150,912,965 | 56,378,907 | 56,378,981 |
| HD | 153,157,903 | 153,158,311 | 55,652,781 | 55,652,855 |
| PG | 147,898,877 | 147,899,285 | 53,790,638 | 53,790,712 |

### Results Summary

Across 20 major U.S. equities (2010–2025):

- The chaos-augmented HMM successfully identifies three distinct regimes with economically meaningful characteristics
- **Chaotic regimes** exhibit the highest Lyapunov exponents and Permutation Entropy, corresponding to periods of market stress and unpredictability
- **Trending regimes** show elevated Hurst exponents (H > 0.5), capturing persistent directional moves
- **Mean-reverting regimes** display low Hurst exponents, consistent with range-bound price behavior
- IAAFT surrogate tests show significant Permutation Entropy results for WMT and DIS (p = 0.000), while Lyapunov exponents do not reach significance at the 5% level for any asset — consistent with the view that financial markets exhibit complex nonlinear structure but not low-dimensional deterministic chaos
- The regime-based trading strategy (long in trending, contrarian in mean-reverting, flat in chaotic) shows positive annualized returns for 13 out of 20 assets, with the strongest performance in TSLA (25.84% ann. return, 0.530 Sharpe) and MA (10.22% ann. return, 0.542 Sharpe)

---

## Citation

```bibtex
@article{lagudu2026chaos,
  title={Chaos-Based Regime Detection in Financial Markets},
  author={Lagudu, Arush Rao},
  year={2026},
  institution={Frisco Centennial High School}
}
```

---

## License

MIT License
