"""
Unified Copper Dataset Builder
------------------------------
Data pipeline to download, align, and convert a globally diversified 
universe of copper-related assets into a unified USD-denominated dataset.

This script handles:
- Bulk downloading of historical OHLCV data via Yahoo Finance.
- Foreign Exchange (FX) rate synchronization and currency conversion.
- Data alignment and missing value imputation (forward-filling).
- Dynamic winsorization of returns to handle outliers without look-ahead bias.
- Exporting structured CSVs for backtesting and statistical analysis.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION & CURRENCY MAPPING
# =============================================================================

# Maps asset suffixes to their respective FX ticker, operation, and scaling divisor
FX_MAPPING = {
    ".TO": ("CAD=X", "div", 1.0),      # CAD to USD
    ".L":  ("GBP=X", "mul", 100.0),    # GBX (Pence) to GBP to USD
    ".WA": ("PLN=X", "div", 1.0),      # PLN to USD
    ".DE": ("EUR=X", "div", 1.0),      # EUR to USD
    "DEFAULT": (None, "none", 1.0)     # USD natively (No conversion needed)
}

def get_copper_heavy_universe() -> Tuple[List[str], List[str]]:
    """
    Defines the copper asset universe and extracts required FX tickers.
    
    Returns:
        Tuple[List[str], List[str]]: A tuple containing the deduplicated list 
        of asset tickers and the list of required FX tickers.
    """
    primary_anchor = ["HG=F"]
    sector_proxies = ["COPX", "CPER"]

    copper_heavy = [
        "FCX", "SCCO", "HBM", "ERO", "TECK", "GLNCY", "VALE", "BHP", "RIO",
        "FM.TO", "CS.TO", "IVN.TO",
        "ANTO.L", "KGH.WA", "NDA.DE",
        "LUNMF", "TGB",
    ]
    
    # Deduplicate the combined universe to prevent redundant downloads
    unique_assets = list(dict.fromkeys(primary_anchor + sector_proxies + copper_heavy))

    # Identify required FX pairs dynamically based on asset suffixes
    fx_tickers = set()
    for t in unique_assets:
        for suffix, (fx, _, _) in FX_MAPPING.items():
            if t.endswith(suffix) and fx is not None:
                fx_tickers.add(fx)
    
    return unique_assets, list(fx_tickers)


# =============================================================================
# 1) CORE PROCESSING (USD CONVERSION)
# =============================================================================

def process_ohlcv_to_usd(df_ohlcv: pd.DataFrame, df_fx: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Open, High, Low, Close, and Adj Close prices to USD using daily FX rates.
    Trading Volume remains unchanged.
    
    Args:
        df_ohlcv: MultiIndex DataFrame containing raw asset price data.
        df_fx: DataFrame containing daily FX closing rates.
        
    Returns:
        pd.DataFrame: A unified MultiIndex DataFrame denominated entirely in USD.
    """
    # Temporal alignment (Strict intersection of valid trading dates)
    common_index = df_ohlcv.index.intersection(df_fx.index)
    df_ohlcv = df_ohlcv.loc[common_index].copy()
    
    # Forward-fill FX rates to handle asynchronous market holidays
    df_fx_aligned = df_fx.loc[common_index].ffill() 

    tickers = df_ohlcv.columns.levels[0]
    print("[PROCESSING] Converting Asset prices to USD...")

    for ticker in tickers:
        # Determine the target currency and conversion logic
        fx_ticker, op, divisor = None, None, 1.0
        match_found = False
        
        for suffix, (fx, operation, div) in FX_MAPPING.items():
            if ticker.endswith(suffix):
                fx_ticker, op, divisor = fx, operation, div
                match_found = True
                break
        
        # Skip assets that are already in USD or lack a valid mapping
        if not match_found or fx_ticker is None:
            continue
            
        if fx_ticker not in df_fx_aligned.columns:
            continue

        rate = df_fx_aligned[fx_ticker]
        
        # Target price fields for conversion
        price_fields = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        
        for field in price_fields:
            if (ticker, field) in df_ohlcv.columns:
                col_data = df_ohlcv[(ticker, field)]
                
                if op == "div":
                    # Example: CAD / Rate = USD
                    df_ohlcv[(ticker, field)] = (col_data / divisor) / rate
                elif op == "mul":
                    # Example: (GBX / 100) * Rate = USD
                    df_ohlcv[(ticker, field)] = (col_data / divisor) * rate

    return df_ohlcv

# =============================================================================
# 2) MAIN PIPELINE
# =============================================================================

def build_unified_dataset(
    period: str = "10y",
    out_ohlc: str = "copper_prices_ohlc_usd.csv",
    out_close: str = "copper_prices_close_usd.csv",
    out_returns: str = "copper_returns_clean.csv"
):
    """
    Executes the complete data extraction, transformation, and loading (ETL) pipeline.
    
    Args:
        period: Time horizon to fetch from Yahoo Finance (e.g., "10y", "5y").
        out_ohlc: Filepath for the comprehensive OHLCV export.
        out_close: Filepath for the Adjusted Close-only export.
        out_returns: Filepath for the cleaned returns export.
    """
    print(f"[INIT] Starting dataset construction (period={period})")
    
    # 1. Data Download
    assets, fx_tickers = get_copper_heavy_universe()
    
    print(f"   -> Downloading Asset History (OHLCV)...")
    data_assets = yf.download(
        assets, period=period, interval="1d", 
        auto_adjust=False, group_by="ticker", progress=False
    )
    
    print(f"   -> Downloading FX Rates (Close)...")
    data_fx = yf.download(
        fx_tickers, period=period, interval="1d", 
        auto_adjust=False, group_by="column", progress=False
    )["Adj Close"]
    
    # 2. USD Unification
    df_full_usd = process_ohlcv_to_usd(data_assets, data_fx)
    
    # 3. Data Cleansing & Alignment
    adj_close_cols = [c for c in df_full_usd.columns if c[1] == 'Adj Close']
    df_check = df_full_usd[adj_close_cols]
    
    # Drop rows where more than 50% of the universe is missing data
    valid_indices = df_check.dropna(thresh=int(len(adj_close_cols) * 0.5)).index
    df_full_usd = df_full_usd.loc[valid_indices].ffill().dropna()
    
    print(f"[DATA] Final Aligned Master Dataset Shape: {df_full_usd.shape[0]} rows")

    # ---------------------------------------------------------
    # EXPORT 1: FULL OHLC (Designed for Backtesting Engines)
    # ---------------------------------------------------------
    # Flatten MultiIndex to Ticker_Field format (e.g., FM.TO_Open)
    df_ohlc_export = df_full_usd.copy()
    df_ohlc_export.columns = [f"{t}_{f}" for t, f in df_ohlc_export.columns]
    
    df_ohlc_export.to_csv(out_ohlc)
    print(f"[EXPORT] OHLC Dataset saved to: {out_ohlc}")

    # ---------------------------------------------------------
    # EXPORT 2: CLOSE ONLY (Designed for Correlation/Cointegration Analysis)
    # ---------------------------------------------------------
    # Isolate 'Adj Close' to reflect true economic value in USD
    df_close_export = df_full_usd.xs('Adj Close', level=1, axis=1).copy()
    
    df_close_export.to_csv(out_close)
    print(f"[EXPORT] Adjusted Close Dataset saved to: {out_close}")

    # ---------------------------------------------------------
    # EXPORT 3: RETURNS (Designed for Risk Modeling & Volatility Profiling)
    # ---------------------------------------------------------
    # Calculate daily returns and apply an expanding-window winsorization
    # to handle extreme outliers without introducing look-ahead bias.
    rets = df_close_export.pct_change()
    lower = rets.expanding(min_periods=252).quantile(0.01)
    upper = rets.expanding(min_periods=252).quantile(0.99)
    rets_clean = rets.clip(lower=lower, upper=upper, axis=1).dropna()
    
    rets_clean.to_csv(out_returns)
    print(f"[EXPORT] Clean Returns Dataset saved to: {out_returns}")

    # Generate a visual integrity report
    plot_check(df_close_export, "copper_data_check.png")

def plot_check(df_close: pd.DataFrame, filename: str):
    """
    Generates a normalized (Base-100) performance chart to visually 
    verify the integrity and coherence of the USD-converted price series.
    
    Args:
        df_close: DataFrame containing unified USD closing prices.
        filename: Output path for the generated PNG chart.
    """
    df_norm = df_close / df_close.iloc[0] * 100
    plt.figure(figsize=(10, 6))
    
    for col in df_norm.columns:
        # Highlight the primary commodity anchor (Copper Futures)
        lw = 2 if "HG=F" in col else 1
        alpha = 1 if "HG=F" in col else 0.4
        plt.plot(df_norm.index, df_norm[col], label=col if "HG=F" in col else "", linewidth=lw, alpha=alpha)
        
    plt.title("Data Integrity Check: Copper Universe (USD Base 100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    build_unified_dataset()
