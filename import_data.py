# 01_build_dataset.py
from __future__ import annotations

import argparse
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# 0) UNIVERSE DEFINITION (Primary Anchor + Sector Proxies + Equities)
# =============================================================================

def get_copper_heavy_universe() -> List[str]:
    """
    Defines the financial universe focused on copper assets.
    
    Includes the primary futures anchor, sector ETFs (proxies), 
    and a selection of global copper mining equities.
    
    Returns:
        List[str]: A deduplicated list of tickers.
    """
    primary_anchor = ["HG=F"]  # Copper futures
    sector_proxies = ["COPX", "CPER"]

    # Selection of major global copper producers and explorers
    copper_heavy = [
        "FCX", "SCCO", "HBM", "ERO", "TECK", "GLNCY", "VALE", "BHP", "RIO",
        "FM.TO", "CS.TO", "IVN.TO",
        "ANTO.L", "KGH.WA", "NDA.DE",
        "LUNMF", "TGB",
    ]

    all_tickers = primary_anchor + sector_proxies + copper_heavy

    # Deduplicate while preserving original order
    seen = set()
    unique: List[str] = []
    for t in all_tickers:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


# =============================================================================
# 1) DATA ACQUISITION AND PROCESSING
# =============================================================================

def _download_ohlcv_yfinance(
    tickers: List[str],
    period: str,
    interval: str,
) -> pd.DataFrame:
    """
    Internal helper to download OHLCV panel data via Yahoo Finance.
    
    Args:
        tickers: List of financial symbols.
        period: Data lookback period (e.g., '10y').
        interval: Data frequency (e.g., '1d').
        
    Returns:
        pd.DataFrame: Raw multi-index or single-index DataFrame.
    """
    if not tickers:
        raise ValueError("Ticker list is empty.")

    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if data is None or data.empty:
        raise ValueError("Download failed or returned empty dataset.")
    return data

def build_and_save_ohlcv_full_dataset(
    tickers: List[str],
    period: str = "10y",
    interval: str = "1d",
    filename: str = "copper_data_ohlcv_full.csv",
) -> pd.DataFrame:
    """
    Downloads and exports a complete OHLCV dataset in 'Wide' format.
    
    Args:
        tickers: List of assets to download.
        period: Time range.
        interval: Data resolution.
        filename: Destination CSV path.
        
    Returns:
        pd.DataFrame: The processed OHLCV DataFrame.
    """
    print(f"\n[DATA] Downloading OHLCV data for {len(tickers)} tickers...")
    data = _download_ohlcv_yfinance(tickers=tickers, period=period, interval=interval)

    if not isinstance(data.columns, pd.MultiIndex):
        # Single ticker case
        t = tickers[0]
        df_ohlcv = pd.DataFrame(index=data.index)
        for f in ["Open", "High", "Low", "Close", "Volume"]:
            df_ohlcv[f"{t}_{f}"] = data[f]
    else:
        # Multi-ticker case: flatten MultiIndex to 'Ticker_Field' format
        fields = ["Open", "High", "Low", "Close", "Volume"]
        pieces = []
        for f in fields:
            tmp = data[f].copy()
            tmp.columns = [f"{c}_{f}" for c in tmp.columns]
            pieces.append(tmp)
        df_ohlcv = pd.concat(pieces, axis=1).sort_index()

    df_ohlcv = df_ohlcv.replace([np.inf, -np.inf], np.nan).sort_index()
    df_ohlcv.to_csv(filename)
    print(f"[DATA] Full OHLCV dataset saved: {filename}")
    return df_ohlcv

def build_and_save_close_dataset(
    tickers: List[str],
    period: str = "10y",
    interval: str = "1d",
    filename: str = "close.csv",
) -> pd.DataFrame:
    """
    Extracts and saves raw 'Close' prices. 
    Designed for UI applications (e.g., PyQt6) requiring unnormalized values.
    """
    print(f"\n[DATA] Extracting raw Closing prices...")
    data = _download_ohlcv_yfinance(tickers=tickers, period=period, interval=interval)

    if isinstance(data.columns, pd.MultiIndex):
        df_close = data["Close"].copy()
    else:
        t = tickers[0]
        df_close = pd.DataFrame({t: data["Close"]})

    # Cleaning: Forward/Backward fill gaps in trading dates
    df_close = df_close.dropna(axis=1, how="all").ffill().bfill()
    df_close = df_close.reindex(columns=sorted(df_close.columns))

    df_close.to_csv(filename)
    print(f"[DATA] 'Close' dataset saved: {os.path.abspath(filename)}")
    return df_close

def build_and_save_clean_dataset(
    tickers: List[str],
    period: str = "10y",
    interval: str = "1d",
    filename: str = "copper_data_clean.csv",
    corr_out_png: str = "copper_universe_correlation_heatmap.png",
    corr_out_csv: str = "copper_universe_correlation_matrix.csv",
    perf_out_png: str = "copper_analysis.png",
) -> pd.DataFrame:
    """
    Constructs a 'Clean' dataset with Base-100 normalization and Winsorization.
    Performs visual analysis (correlation heatmap and performance charts).
    """
    data = _download_ohlcv_yfinance(tickers=tickers, period=period, interval=interval)

    if isinstance(data.columns, pd.MultiIndex):
        df_close = data["Close"].copy()
    else:
        t = tickers[0]
        df_close = pd.DataFrame({t: data["Close"]})

    df_close = df_close.dropna(axis=1, how="all").ffill().bfill()

    # Generate Visualizations
    plot_universe_correlation_heatmap(df_close, corr_out_png, corr_out_csv)
    
    # Base-100 Performance Plot (using Adjusted Close if available)
    df_adj = data["Adj Close"].copy().ffill().bfill() if "Adj Close" in data.columns else df_close
    plot_normalized_base100_performance(df_adj, perf_out_png)

    # Return Winsorization (Managing Outliers at 1% and 99% quantiles)
    returns = df_close.pct_change()
    returns_cleaned = returns.clip(lower=returns.quantile(0.01), upper=returns.quantile(0.99), axis=1)

    # Reconstruct Base-100 Normalized Index
    df_clean_norm = (1.0 + returns_cleaned.fillna(0.0)).cumprod() * 100.0
    df_clean_norm.to_csv(filename)
    print(f"[DATA] Clean (Base-100) dataset saved: {filename}")
    return df_clean_norm

# =============================================================================
# 2) ANALYSIS AND VISUALIZATION
# =============================================================================

def plot_universe_correlation_heatmap(df, out_png, out_csv, title="Correlation Matrix"):
    """Generates and saves a heatmap of asset return correlations."""
    rets = df.pct_change().dropna(how="all")
    corr = rets.corr()
    corr.to_csv(out_csv)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", annot_kws={"size": 8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_normalized_base100_performance(df, out_png, title="Base-100 Performance", max_final_index=800.0):
    """Plots normalized growth of assets, highlighting the Copper Futures anchor."""
    df_norm = (df.divide(df.iloc[0].replace(0.0, np.nan), axis=1)) * 100.0
    
    # Filter out extreme outliers/anomalies for chart readability
    keep_cols = [c for c in df_norm.columns if c == "HG=F" or (df_norm[c].iloc[-1] <= max_final_index)]
    df_norm = df_norm[keep_cols]

    plt.figure(figsize=(12, 6))
    for col in df_norm.columns:
        # Highlight HG=F (Copper Futures) with a thicker black line
        alpha, lw = (1.0, 2.6) if col == "HG=F" else (0.6, 0.9)
        color = "black" if col == "HG=F" else None
        plt.plot(df_norm.index, df_norm[col], linewidth=lw, alpha=alpha, color=color, label=col if col=="HG=F" else "")
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, dpi=180)
    plt.close()

# =============================================================================
# 3) CLI AND MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution entry point."""
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Copper Market Data Engineering Pipeline.")
    parser.add_argument("--period", type=str, default="10y", help="Lookback period (e.g., 5y, 10y, max)")
    parser.add_argument("--out_full", type=str, default="copper_data_ohlcv_full.csv", help="Full OHLCV filename")
    parser.add_argument("--out_clean", type=str, default="copper_data_clean.csv", help="Normalized dataset filename")
    parser.add_argument("--out_close", type=str, default="close.csv", help="Raw Close price filename")
    args = parser.parse_args()

    tickers = get_copper_heavy_universe()

    # Phase 1: Full OHLCV Dataset
    build_and_save_ohlcv_full_dataset(tickers, args.period, "1d", args.out_full)

    # Phase 2: Raw Close Prices (for GUI applications)
    build_and_save_close_dataset(tickers, args.period, "1d", args.out_close)

    # Phase 3: Clean Normalized Dataset (Base-100 + Winsorization)
    build_and_save_clean_dataset(tickers, args.period, "1d", args.out_clean)

    print("\n[SUCCESS] Pipeline execution complete.")

if __name__ == "__main__":
    main()
