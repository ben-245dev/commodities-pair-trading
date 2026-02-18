"""
Unified Copper Dataset Builder
------------------------------
This script constructs a clean, USD-normalized dataset for copper-related assets.
It handles:
1. Multi-currency OHLCV downloading (CAD, GBP, PLN, EUR, USD).
2. Dynamic FX conversion to a base currency (USD).
3. Data cleaning and time-alignment across different international exchanges.
4. Outlier removal (Winsorization) using expanding windows to avoid look-ahead bias.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Configuration for Currency Conversion
# Format: Suffix: (FX_Ticker, Operation, Unit_Divisor)
FX_MAPPING = {
    ".TO": ("CAD=X", "div", 1.0),    # CAD price / (CAD/USD rate) = USD
    ".L":  ("GBP=X", "mul", 0.01),   # (GBX price * 0.01) * (GBP/USD rate) = USD
    ".WA": ("PLN=X", "div", 1.0),    # PLN price / (PLN/USD rate) = USD
    ".DE": ("EUR=X", "div", 1.0),    # EUR price / (EUR/USD rate) = USD
    "DEFAULT": (None, "none", 1.0)
}

def get_copper_heavy_universe() -> Tuple[List[str], List[str]]:
    """
    Defines the investment universe: Copper Futures, ETFs, and Global Miners.
    Returns a tuple of (Asset Tickers, Required FX Tickers).
    """
    primary_anchor = ["HG=F"] # Copper High Grade Futures
    sector_proxies = ["COPX", "CPER"]

    # Global miners across different exchanges
    copper_heavy = [
        "FCX", "SCCO", "HBM", "ERO", "TECK", "GLNCY", "VALE", "BHP", "RIO",
        "FM.TO", "CS.TO", "IVN.TO",
        "ANTO.L", "KGH.WA", "NDA.DE",
        "LUNMF", "TGB",
    ]
    
    unique_assets = list(dict.fromkeys(primary_anchor + sector_proxies + copper_heavy))

    # Identify which FX rates we need to download based on the suffixes
    fx_tickers = set()
    for t in unique_assets:
        for suffix, (fx, _, _) in FX_MAPPING.items():
            if t.endswith(suffix) and fx is not None:
                fx_tickers.add(fx)
    
    return unique_assets, list(fx_tickers)


def process_ohlcv_to_usd(df_ohlcv: pd.DataFrame, df_fx: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Open, High, Low, Close, and Adj Close columns to USD.
    Volume remains unchanged.
    """
    # Align dates across all datasets
    common_index = df_ohlcv.index.intersection(df_fx.index)
    df_ohlcv = df_ohlcv.loc[common_index].copy()
    df_fx_aligned = df_fx.loc[common_index].ffill() 

    tickers = df_ohlcv.columns.levels[0]
    print("[PROCESSING] Normalizing all assets to USD base...")

    for ticker in tickers:
        fx_ticker, op, unit_adj = None, None, 1.0
        match_found = False
        
        # Determine conversion rule
        for suffix, (fx, operation, adj) in FX_MAPPING.items():
            if ticker.endswith(suffix):
                fx_ticker, op, unit_adj = fx, operation, adj
                match_found = True
                break
        
        if not match_found or fx_ticker is None:
            continue
            
        if fx_ticker not in df_fx_aligned.columns:
            warnings.warn(f"FX Rate {fx_ticker} missing for {ticker}. Skipping conversion.")
            continue

        rate = df_fx_aligned[fx_ticker]
        price_fields = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        
        for field in price_fields:
            if (ticker, field) in df_ohlcv.columns:
                series = df_ohlcv[(ticker, field)]
                
                # Apply unit adjustment (e.g., Pence to Pounds) and then FX rate
                if op == "div":
                    df_ohlcv[(ticker, field)] = (series * unit_adj) / rate
                elif op == "mul":
                    df_ohlcv[(ticker, field)] = (series * unit_adj) * rate

    return df_ohlcv


def build_unified_dataset(
    period: str = "10y",
    out_ohlc: str = "copper_prices_ohlc_usd.csv",
    out_close: str = "copper_prices_close_usd.csv",
    out_returns: str = "copper_returns_clean.csv"
):
    """Main pipeline to download, process, and save the datasets."""
    print(f"[INIT] Building dataset for period: {period}")
    
    # 1. Data Acquisition
    assets, fx_tickers = get_copper_heavy_universe()
    
    print(f" -> Downloading {len(assets)} Assets (OHLCV)...")
    data_assets = yf.download(
        assets, period=period, interval="1d", 
        auto_adjust=False, group_by="ticker", progress=False
    )
    
    print(f" -> Downloading {len(fx_tickers)} FX Rates...")
    data_fx = yf.download(
        fx_tickers, period=period, interval="1d", 
        auto_adjust=False, progress=False
    )["Adj Close"]
    
    # 2. Currency Normalization
    df_full_usd = process_ohlcv_to_usd(data_assets, data_fx)
    
    # 3. Data Cleaning (Alignment & Handling Missing Days)
    # Filter for rows where at least 50% of assets have valid data
    adj_close_cols = [c for c in df_full_usd.columns if c[1] == 'Adj Close']
    valid_rows = df_full_usd[adj_close_cols].dropna(thresh=int(len(adj_close_cols) * 0.5)).index
    
    # Finalize dataset with forward fill for minor gaps (holidays)
    df_full_usd = df_full_usd.loc[valid_rows].ffill().dropna()
    print(f"[DATA] Final cleaned shape: {df_full_usd.shape}")

    # 4. EXPORT: Full OHLC (for backtesting)
    df_ohlc_export = df_full_usd.copy()
    df_ohlc_export.columns = [f"{t}_{f}" for t, f in df_ohlc_export.columns]
    df_ohlc_export.to_csv(out_ohlc)

    # 5. EXPORT: Adjusted Close Only (for correlation/analysis)
    df_close_export = df_full_usd.xs('Adj Close', level=1, axis=1).copy()
    df_close_export.to_csv(out_close)

    # 6. EXPORT: Cleaned Returns
    # Use expanding window for Winsorization to prevent look-ahead bias
    rets = df_close_export.pct_change()
    lower = rets.expanding(min_periods=252).quantile(0.01)
    upper = rets.expanding(min_periods=252).quantile(0.99)
    rets_clean = rets.clip(lower=lower, upper=upper, axis=1).dropna()
    rets_clean.to_csv(out_returns)

    print(f"[SUCCESS] Exports completed: {out_ohlc}, {out_close}, {out_returns}")

    # Visual Quality Check
    plot_check(df_close_export, "copper_data_integrity.png")


def plot_check(df_close: pd.DataFrame, filename: str):
    """Generates a normalized price chart to verify data consistency."""
    df_norm = (df_close / df_close.iloc[0]) * 100
    plt.figure(figsize=(12, 7))
    
    for col in df_norm.columns:
        is_copper = "HG=F" in col
        plt.plot(
            df_norm.index, df_norm[col], 
            label=col if is_copper else "", 
            linewidth=2.5 if is_copper else 1.0, 
            alpha=1.0 if is_copper else 0.4,
            color="chocolate" if is_copper else None
        )
        
    plt.title("Asset Integrity Check (USD Base 100 Normalized)")
    plt.ylabel("Relative Growth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    build_unified_dataset()
