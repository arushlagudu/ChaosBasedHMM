"""Data loading and preprocessing for chaos-based regime detection."""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def download_price(symbol, start=None, end=None, interval='1d'):
    """Download historical adjusted close price using yfinance."""
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices = df['Close'].rename('close').to_frame()
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices


def compute_returns(prices: pd.Series):
    return np.log(prices).diff().rename('ret')


def prepare_timeseries(symbol, start=None, end=None, interval='1d'):
    df = download_price(symbol, start, end, interval)
    df['ret'] = compute_returns(df['close'])
    df = df.dropna().copy()
    # Winsorize extreme returns (beyond 10 sigma)
    mu, sigma = df['ret'].mean(), df['ret'].std()
    df['ret'] = df['ret'].clip(mu - 10 * sigma, mu + 10 * sigma)
    return df
