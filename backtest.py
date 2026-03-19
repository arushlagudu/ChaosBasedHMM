"""Regime-based backtesting framework."""
import numpy as np
import pandas as pd


def regime_strategy(prices_df, states_series, mapping,
                    initial_capital=100000.0, tc=0.0005):
    """Simple regime-based trading strategy.
    - Trending: long
    - Mean-reverting: contrarian based on return sign
    - Chaotic: flat (cash)
    """
    df = prices_df.copy()
    df['state'] = states_series
    df['regime'] = df['state'].map(mapping)
    n = len(df)
    positions = np.zeros(n)
    for t in range(n):
        regime = df['regime'].iloc[t]
        if regime == 'trending':
            positions[t] = 1.0
        elif regime == 'mean_reverting':
            r = df['ret'].iloc[t]
            if r < -0.005:
                positions[t] = 1.0
            elif r > 0.005:
                positions[t] = -1.0
            else:
                positions[t] = 0.0
        else:  # chaotic or NaN
            positions[t] = 0.0
    # Shift positions by 1 (trade on signal, earn next return)
    df['position'] = pd.Series(positions, index=df.index).shift(1).fillna(0)
    df['next_ret'] = df['ret'].shift(-1).fillna(0)
    # Transaction costs
    df['trade'] = df['position'].diff().abs().fillna(0)
    df['strategy_ret'] = df['position'] * df['next_ret'] - df['trade'] * tc
    df['nav'] = initial_capital * (1 + df['strategy_ret']).cumprod()
    # Buy and hold benchmark
    df['bh_ret'] = df['ret'].shift(-1).fillna(0)
    df['bh_nav'] = initial_capital * (1 + df['bh_ret']).cumprod()
    return df


def evaluate_backtest(df, nav_col='nav'):
    """Compute performance metrics."""
    nav = df[nav_col].dropna()
    if len(nav) < 10:
        return {}
    returns = nav.pct_change().dropna()
    ann_ret = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum_max = nav.cummax()
    drawdown = (cum_max - nav) / cum_max
    max_dd = drawdown.max()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    return dict(
        total_return=total_ret,
        annual_return=ann_ret,
        annual_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )


def baseline_vol_strategy(prices_df, vol_col='vol_short',
                          initial_capital=100000.0, tc=0.0005):
    """Baseline: volatility-only HMM with 2 states (high/low vol)."""
    from model import RegimeHMM
    feature_cols = ['ret', vol_col]
    hmm = RegimeHMM(n_states=2, cov_type='diag', random_state=42)
    sub = prices_df.dropna(subset=feature_cols)
    hmm.fit(sub, feature_cols)
    states = hmm.predict_states(sub)
    # Map: lower vol state -> invest, higher vol state -> cash
    vol_means = {}
    for s in range(2):
        idx = (states == s)
        if idx.sum() > 0:
            vol_means[s] = sub.loc[idx, vol_col].mean()
        else:
            vol_means[s] = np.inf
    s_low = min(vol_means, key=vol_means.get)
    s_high = max(vol_means, key=vol_means.get)
    mapping = {s_low: 'trending', s_high: 'chaotic'}  # invest vs cash
    return regime_strategy(sub, states, mapping, initial_capital, tc)
