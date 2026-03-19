"""Main analysis pipeline for chaos-based regime detection."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

from data_loader import prepare_timeseries
from features import compute_features_df
from model import RegimeHMM, map_states_to_regimes
from backtest import regime_strategy, evaluate_backtest, baseline_vol_strategy
from surrogates import surrogate_test_lyapunov, surrogate_test_perm_entropy

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif'
})

SYMBOLS = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'BAC', 'XOM',
           'JNJ', 'PFE', 'WMT', 'DIS', 'TSLA', 'NVDA', 'META', 'V',
           'MA', 'UNH', 'HD', 'PG']
START = '2010-01-01'
END = '2025-03-19'
FEATURE_COLS = ['ret', 'vol_short', 'lyap', 'hurst', 'perm_entropy']
BASELINE_COLS = ['ret', 'vol_short']

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
TABLES_DIR = 'tables'

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)


def run_single_asset(symbol, start=START, end=END):
    """Run complete pipeline for one asset."""
    print(f"\n{'='*60}")
    print(f"Processing {symbol}")
    print(f"{'='*60}")

    # 1. Download data
    print(f"[{symbol}] Downloading data...")
    try:
        df = prepare_timeseries(symbol, start=start, end=end)
    except Exception as e:
        print(f"[{symbol}] Failed to download: {e}")
        return None
    print(f"[{symbol}] Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} days")

    # 2. Compute features
    print(f"[{symbol}] Computing features...")
    df = compute_features_df(df, w_short=30, w_med=125, w_long=250,
                             emb_dim=3, tau=1, perm_m=5, verbose=True)

    # 3. Fit HMM (chaos-augmented)
    print(f"[{symbol}] Fitting chaos-augmented HMM...")
    sub = df.dropna(subset=FEATURE_COLS).copy()
    if len(sub) < 100:
        print(f"[{symbol}] Insufficient data after feature computation")
        return None

    hmm = RegimeHMM(n_states=3, cov_type='full', random_state=42)
    hmm.fit(sub, FEATURE_COLS)
    states = hmm.predict_states(sub)
    mapping, state_stats = map_states_to_regimes(hmm, sub)
    sub['state'] = states
    sub['regime'] = sub['state'].map(mapping)
    aic, bic = hmm.get_aic_bic(sub)

    # 4. Fit baseline HMM (returns + vol only)
    print(f"[{symbol}] Fitting baseline HMM...")
    hmm_base = RegimeHMM(n_states=2, cov_type='diag', random_state=42)
    sub_base = sub.dropna(subset=BASELINE_COLS)
    hmm_base.fit(sub_base, BASELINE_COLS)
    aic_base, bic_base = hmm_base.get_aic_bic(sub_base)

    # 5. Regime statistics
    print(f"[{symbol}] Computing regime statistics...")
    regime_stats = {}
    for regime_name in ['chaotic', 'trending', 'mean_reverting']:
        mask = sub['regime'] == regime_name
        if mask.sum() < 5:
            continue
        rs = sub.loc[mask]
        next_ret = sub['ret'].shift(-1)
        regime_stats[regime_name] = {
            'count': int(mask.sum()),
            'pct': float(mask.sum() / len(sub)),
            'mean_ret': float(rs['ret'].mean()),
            'std_ret': float(rs['ret'].std()),
            'mean_vol': float(rs['vol_short'].mean()),
            'mean_lyap': float(rs['lyap'].mean()),
            'mean_hurst': float(rs['hurst'].mean()),
            'mean_pe': float(rs['perm_entropy'].mean()),
            'mean_next_ret': float(next_ret[mask].mean()),
            'std_next_ret': float(next_ret[mask].std()),
        }

    # 6. Transition matrix
    trans_mat = hmm.get_transition_matrix()

    # 7. Backtest
    print(f"[{symbol}] Running backtest...")
    bt_df = regime_strategy(sub, states, mapping, tc=0.0005)
    perf = evaluate_backtest(bt_df, nav_col='nav')
    perf_bh = evaluate_backtest(bt_df, nav_col='bh_nav')

    # 8. Surrogate testing (on a representative window)
    print(f"[{symbol}] Running surrogate tests (19 surrogates for speed)...")
    mid = len(sub) // 2
    test_window = sub['ret'].values[max(0, mid-250):mid+250]
    lyap_obs, lyap_surr_mean, lyap_pval, lyap_sig = surrogate_test_lyapunov(
        test_window, n_surrogates=19, emb_dim=3, tau=1, k_max=50)
    pe_obs, pe_surr_mean, pe_pval, pe_sig = surrogate_test_perm_entropy(
        test_window, n_surrogates=19, m=5, tau=1)

    surrogate_results = {
        'lyap_obs': float(lyap_obs) if not np.isnan(lyap_obs) else None,
        'lyap_surr_mean': float(lyap_surr_mean) if not np.isnan(lyap_surr_mean) else None,
        'lyap_pval': float(lyap_pval) if not np.isnan(lyap_pval) else None,
        'lyap_significant': bool(lyap_sig),
        'pe_obs': float(pe_obs) if not np.isnan(pe_obs) else None,
        'pe_surr_mean': float(pe_surr_mean) if not np.isnan(pe_surr_mean) else None,
        'pe_pval': float(pe_pval) if not np.isnan(pe_pval) else None,
        'pe_significant': bool(pe_sig),
    }

    # 9. Statistical tests: next-day returns differ across regimes?
    regime_groups = {}
    next_ret = sub['ret'].shift(-1).dropna()
    for regime_name in ['chaotic', 'trending', 'mean_reverting']:
        mask = (sub['regime'] == regime_name).reindex(next_ret.index, fill_value=False)
        vals = next_ret[mask].values
        if len(vals) > 5:
            regime_groups[regime_name] = vals
    stat_tests = {}
    regimes_list = list(regime_groups.keys())
    for i in range(len(regimes_list)):
        for j in range(i+1, len(regimes_list)):
            r1, r2 = regimes_list[i], regimes_list[j]
            tstat, pval = ttest_ind(regime_groups[r1], regime_groups[r2])
            stat_tests[f'{r1}_vs_{r2}'] = {'t_stat': float(tstat), 'p_value': float(pval)}

    # 10. Dwell times
    dwell_times = {}
    for regime_name in ['chaotic', 'trending', 'mean_reverting']:
        mask = (sub['regime'] == regime_name).values
        runs = []
        count = 0
        for v in mask:
            if v:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        if runs:
            dwell_times[regime_name] = {'mean': float(np.mean(runs)),
                                         'median': float(np.median(runs)),
                                         'max': int(np.max(runs))}

    result = {
        'symbol': symbol,
        'n_obs': len(sub),
        'date_range': f"{sub.index[0].date()} to {sub.index[-1].date()}",
        'mapping': {str(k): v for k, v in mapping.items()},
        'state_means': {str(k): {kk: float(vv) for kk, vv in v.items()} if v is not None else None
                        for k, v in state_stats.items()},
        'regime_stats': regime_stats,
        'transition_matrix': trans_mat.tolist(),
        'aic': float(aic), 'bic': float(bic),
        'aic_baseline': float(aic_base), 'bic_baseline': float(bic_base),
        'backtest_chaos': perf,
        'backtest_buyhold': perf_bh,
        'surrogate_tests': surrogate_results,
        'stat_tests': stat_tests,
        'dwell_times': dwell_times,
    }

    # Save per-asset data
    sub.to_csv(os.path.join(RESULTS_DIR, f'{symbol}_features.csv'))

    # Generate plots
    generate_plots(symbol, sub, mapping, bt_df)

    print(f"[{symbol}] Done. Regime distribution: {sub['regime'].value_counts().to_dict()}")
    return result


def generate_plots(symbol, sub, mapping, bt_df):
    """Generate per-asset figures."""
    colors = {'chaotic': '#E74C3C', 'trending': '#2ECC71', 'mean_reverting': '#3498DB'}

    # 1. Regime overlay on price
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    ax = axes[0]
    ax.plot(sub.index, sub['close'], 'k-', lw=0.8, alpha=0.9)
    for regime_name, color in colors.items():
        mask = sub['regime'] == regime_name
        if mask.any():
            for idx in sub.index[mask]:
                ax.axvspan(idx, idx + pd.Timedelta(days=1),
                          alpha=0.3, color=color, linewidth=0)
    ax.set_ylabel('Price')
    ax.set_title(f'{symbol}: Price with Detected Regimes')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.5, label=r)
                       for r, c in colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    axes[1].plot(sub.index, sub['lyap'], 'r-', lw=0.5, alpha=0.8)
    axes[1].set_ylabel('Lyapunov')
    axes[1].axhline(0, color='gray', ls='--', lw=0.5)

    axes[2].plot(sub.index, sub['hurst'], 'b-', lw=0.5, alpha=0.8)
    axes[2].set_ylabel('Hurst')
    axes[2].axhline(0.5, color='gray', ls='--', lw=0.5)

    axes[3].plot(sub.index, sub['perm_entropy'], 'g-', lw=0.5, alpha=0.8)
    axes[3].set_ylabel('Perm. Entropy')
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'{symbol}_regimes.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, f'{symbol}_regimes.png'))
    plt.close(fig)

    # 2. Backtest equity curves
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt_df.index, bt_df['nav'], 'b-', lw=1, label='Chaos-Regime Strategy')
    ax.plot(bt_df.index, bt_df['bh_nav'], 'k--', lw=1, alpha=0.7, label='Buy & Hold')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'{symbol}: Backtest Equity Curves')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'{symbol}_backtest.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, f'{symbol}_backtest.png'))
    plt.close(fig)


def generate_summary_figures(all_results):
    """Generate cross-asset summary figures."""
    # 1. Regime proportions across assets
    symbols = [r['symbol'] for r in all_results]
    regimes = ['chaotic', 'trending', 'mean_reverting']
    proportions = {r: [] for r in regimes}
    for res in all_results:
        for r in regimes:
            if r in res['regime_stats']:
                proportions[r].append(res['regime_stats'][r]['pct'])
            else:
                proportions[r].append(0.0)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(symbols))
    width = 0.25
    colors = ['#E74C3C', '#2ECC71', '#3498DB']
    for i, (r, c) in enumerate(zip(regimes, colors)):
        ax.bar(x + i * width, proportions[r], width, label=r.replace('_', ' ').title(), color=c, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(symbols, rotation=45, ha='right')
    ax.set_ylabel('Proportion of Days')
    ax.set_title('Regime Distribution Across Assets')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'regime_proportions.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'regime_proportions.png'))
    plt.close(fig)

    # 2. Sharpe ratio comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    sharpe_chaos = [r['backtest_chaos'].get('sharpe', 0) for r in all_results]
    sharpe_bh = [r['backtest_buyhold'].get('sharpe', 0) for r in all_results]
    ax.bar(x - 0.15, sharpe_chaos, 0.3, label='Chaos-Regime Strategy', color='#E74C3C', alpha=0.8)
    ax.bar(x + 0.15, sharpe_bh, 0.3, label='Buy & Hold', color='#3498DB', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Strategy Performance Comparison')
    ax.legend()
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'sharpe_comparison.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'sharpe_comparison.png'))
    plt.close(fig)

    # 3. Chaos metrics box plots by regime (pooled across assets)
    # Load feature CSVs
    all_features = []
    for sym in symbols:
        fp = os.path.join(RESULTS_DIR, f'{sym}_features.csv')
        if os.path.exists(fp):
            tmp = pd.read_csv(fp, index_col=0, parse_dates=True)
            tmp['symbol'] = sym
            all_features.append(tmp)
    if all_features:
        pooled = pd.concat(all_features)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, col, title in zip(axes,
                                    ['lyap', 'hurst', 'perm_entropy'],
                                    ['Lyapunov Exponent', 'Hurst Exponent', 'Permutation Entropy']):
            data = []
            labels = []
            for r in regimes:
                vals = pooled.loc[pooled['regime'] == r, col].dropna()
                if len(vals) > 0:
                    data.append(vals.values)
                    labels.append(r.replace('_', ' ').title())
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                for patch, c in zip(bp['boxes'], ['#E74C3C', '#2ECC71', '#3498DB']):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
            ax.set_title(title)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'feature_boxplots.pdf'))
        fig.savefig(os.path.join(FIGURES_DIR, 'feature_boxplots.png'))
        plt.close(fig)

    # 4. Average transition matrix heatmap
    all_trans = np.array([r['transition_matrix'] for r in all_results])
    avg_trans = np.mean(all_trans, axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(avg_trans, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['State 0', 'State 1', 'State 2'],
                yticklabels=['State 0', 'State 1', 'State 2'], ax=ax)
    ax.set_title('Average Transition Matrix Across Assets')
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'avg_transition_matrix.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'avg_transition_matrix.png'))
    plt.close(fig)


def generate_tables(all_results):
    """Generate summary tables as CSV and LaTeX."""
    # Table 1: Summary statistics per asset
    rows = []
    for r in all_results:
        row = {
            'Asset': r['symbol'],
            'Obs.': r['n_obs'],
            'AIC (Chaos)': f"{r['aic']:.0f}",
            'BIC (Chaos)': f"{r['bic']:.0f}",
            'AIC (Baseline)': f"{r['aic_baseline']:.0f}",
            'BIC (Baseline)': f"{r['bic_baseline']:.0f}",
        }
        for regime in ['chaotic', 'trending', 'mean_reverting']:
            if regime in r['regime_stats']:
                row[f'{regime} %'] = f"{r['regime_stats'][regime]['pct']*100:.1f}"
            else:
                row[f'{regime} %'] = '0.0'
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(TABLES_DIR, 'summary_stats.csv'), index=False)
    df_summary.to_latex(os.path.join(TABLES_DIR, 'summary_stats.tex'), index=False, escape=True)

    # Table 2: Regime characteristics (pooled)
    rows2 = []
    for regime in ['chaotic', 'trending', 'mean_reverting']:
        vals = {'Regime': regime.replace('_', ' ').title()}
        lyaps, hursts, pes, vols, rets = [], [], [], [], []
        for r in all_results:
            if regime in r['regime_stats']:
                lyaps.append(r['regime_stats'][regime]['mean_lyap'])
                hursts.append(r['regime_stats'][regime]['mean_hurst'])
                pes.append(r['regime_stats'][regime]['mean_pe'])
                vols.append(r['regime_stats'][regime]['mean_vol'])
                rets.append(r['regime_stats'][regime]['mean_ret'])
        if lyaps:
            vals['Lyapunov'] = f"{np.mean(lyaps):.4f} ({np.std(lyaps):.4f})"
            vals['Hurst'] = f"{np.mean(hursts):.4f} ({np.std(hursts):.4f})"
            vals['Perm. Entropy'] = f"{np.mean(pes):.4f} ({np.std(pes):.4f})"
            vals['Volatility'] = f"{np.mean(vols):.6f} ({np.std(vols):.6f})"
            vals['Mean Return'] = f"{np.mean(rets):.6f} ({np.std(rets):.6f})"
        rows2.append(vals)
    df_regimes = pd.DataFrame(rows2)
    df_regimes.to_csv(os.path.join(TABLES_DIR, 'regime_characteristics.csv'), index=False)
    df_regimes.to_latex(os.path.join(TABLES_DIR, 'regime_characteristics.tex'), index=False, escape=True)

    # Table 3: Backtest performance
    rows3 = []
    for r in all_results:
        pc = r['backtest_chaos']
        pb = r['backtest_buyhold']
        rows3.append({
            'Asset': r['symbol'],
            'Strategy Return': f"{pc.get('total_return', 0)*100:.1f}%",
            'Strategy Sharpe': f"{pc.get('sharpe', 0):.3f}",
            'Strategy MaxDD': f"{pc.get('max_drawdown', 0)*100:.1f}%",
            'B&H Return': f"{pb.get('total_return', 0)*100:.1f}%",
            'B&H Sharpe': f"{pb.get('sharpe', 0):.3f}",
            'B&H MaxDD': f"{pb.get('max_drawdown', 0)*100:.1f}%",
        })
    df_bt = pd.DataFrame(rows3)
    df_bt.to_csv(os.path.join(TABLES_DIR, 'backtest_results.csv'), index=False)
    df_bt.to_latex(os.path.join(TABLES_DIR, 'backtest_results.tex'), index=False, escape=True)

    # Table 4: Surrogate test results
    rows4 = []
    for r in all_results:
        st = r['surrogate_tests']
        rows4.append({
            'Asset': r['symbol'],
            'Lyap. Obs.': f"{st['lyap_obs']:.4f}" if st['lyap_obs'] is not None else 'N/A',
            'Lyap. Surr.': f"{st['lyap_surr_mean']:.4f}" if st['lyap_surr_mean'] is not None else 'N/A',
            'Lyap. p-val': f"{st['lyap_pval']:.3f}" if st['lyap_pval'] is not None else 'N/A',
            'Lyap. Sig.': 'Yes' if st['lyap_significant'] else 'No',
            'PE Obs.': f"{st['pe_obs']:.4f}" if st['pe_obs'] is not None else 'N/A',
            'PE Surr.': f"{st['pe_surr_mean']:.4f}" if st['pe_surr_mean'] is not None else 'N/A',
            'PE p-val': f"{st['pe_pval']:.3f}" if st['pe_pval'] is not None else 'N/A',
            'PE Sig.': 'Yes' if st['pe_significant'] else 'No',
        })
    df_surr = pd.DataFrame(rows4)
    df_surr.to_csv(os.path.join(TABLES_DIR, 'surrogate_tests.csv'), index=False)
    df_surr.to_latex(os.path.join(TABLES_DIR, 'surrogate_tests.tex'), index=False, escape=True)

    # Table 5: Dwell times
    rows5 = []
    for r in all_results:
        row = {'Asset': r['symbol']}
        for regime in ['chaotic', 'trending', 'mean_reverting']:
            if regime in r['dwell_times']:
                dt = r['dwell_times'][regime]
                row[f'{regime} mean'] = f"{dt['mean']:.1f}"
                row[f'{regime} max'] = f"{dt['max']}"
            else:
                row[f'{regime} mean'] = 'N/A'
                row[f'{regime} max'] = 'N/A'
        rows5.append(row)
    df_dwell = pd.DataFrame(rows5)
    df_dwell.to_csv(os.path.join(TABLES_DIR, 'dwell_times.csv'), index=False)
    df_dwell.to_latex(os.path.join(TABLES_DIR, 'dwell_times.tex'), index=False, escape=True)


if __name__ == '__main__':
    all_results = []
    for sym in SYMBOLS:
        result = run_single_asset(sym)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No results! Exiting.")
        sys.exit(1)

    # Save all results
    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary figures and tables
    print("\n" + "="*60)
    print("Generating summary figures and tables...")
    print("="*60)
    generate_summary_figures(all_results)
    generate_tables(all_results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        pc = r['backtest_chaos']
        pb = r['backtest_buyhold']
        print(f"{r['symbol']:6s} | Chaos Sharpe: {pc.get('sharpe',0):+.3f} | "
              f"B&H Sharpe: {pb.get('sharpe',0):+.3f} | "
              f"Lyap sig: {r['surrogate_tests']['lyap_significant']}")

    print(f"\nTotal assets analyzed: {len(all_results)}")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"Figures saved to {FIGURES_DIR}/")
    print(f"Tables saved to {TABLES_DIR}/")
