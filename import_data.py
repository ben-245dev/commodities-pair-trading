import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Benchmark: Copper Futures
BENCHMARK = "HG=F"

# Expanded Universe - Categorized for better organization
UNIVERSE = {
    # === COPPER-FOCUSED ===
    "Copper ETFs": [
        "COPX",   # Global X Copper Miners ETF (Pure Play)
        "CPER",   # United States Copper Index Fund
        "JJCB",   # iPath Series B Bloomberg Copper ETN (if available)
    ],
    
    # === PURE COPPER MINERS (Primary Revenue >50% from Copper) ===
    "Pure Copper Miners": [
        # North America
        "FCX",    # Freeport-McMoRan (US) - Largest copper producer
        "SCCO",   # Southern Copper Corp (Peru/Mexico)
        "TRQ",    # Turquoise Hill Resources (Mongolia - Oyu Tolgoi)
        "HBM",    # Hudbay Minerals (Canada)
        "CMCL",   # Caledonia Mining Corporation
        "TECK",   # Teck Resources (Canada - diversified but heavy copper)
        
        # International (ADRs/OTC)
        "IVPAF",  # Ivanhoe Mines (Congo - Kamoa-Kakula)
        "LUNMF",  # Lundin Mining (Chile/Sweden)
        "GLNCY",  # Glencore (Switzerland) - OTC
        "CMCLF",  # Codelco (Chile) - if tradeable
        "ERO",    # Ero Copper (Brazil/Canada)
        "CMMC",   # Capstone Copper Corp
        "TGLDF",  # Taseko Mines (Canada/US)
        "SLVRF",  # Silver One Resources (exploration)
    ],
    
    # === DIVERSIFIED MINERS (Copper + Other Metals) ===
    "Diversified Miners": [
        "BHP",    # BHP Group (Australia) - Iron Ore + Copper
        "RIO",    # Rio Tinto (UK/Australia) - Diversified
        "VALE",   # Vale SA (Brazil) - Iron Ore + Nickel + Copper
        "AAL",    # Anglo American (UK) - Copper + Diamonds + Platinum
        "GLNCY",  # Glencore (already listed but important)
        "SCHN",   # Schnitzer Steel (Recycling - copper exposure)
    ],
    
    # === MINING SECTOR ETFs ===
    "Mining ETFs": [
        "PICK",   # iShares MSCI Global Metals & Mining Producers
        "XME",    # SPDR S&P Metals & Mining ETF
        "GDX",    # VanEck Gold Miners (inverse correlation check)
        "GDXJ",   # VanEck Junior Gold Miners
        "SIL",    # Global X Silver Miners (check correlation)
        "REMX",   # VanEck Rare Earth/Strategic Metals ETF
    ],
    
    # === INDUSTRIAL METALS (Correlation Check) ===
    "Related Metals": [
        "ALI=F",  # Aluminum Futures
        "NI=F",   # Nickel Futures
        "ZN=F",   # Zinc Futures
        "PL=F",   # Platinum Futures
    ],
    
    # === COPPER-DEPENDENT INDUSTRIES ===
    "Copper Consumers": [
        # Electrical Equipment (Heavy Copper Users)
        "ETN",    # Eaton Corporation (electrical components)
        "EMR",    # Emerson Electric
        "ROK",    # Rockwell Automation
        "ABB",    # ABB Ltd (Swiss electrical equipment)
        
        # Wire & Cable Manufacturers
        "WCC",    # WESCO International (electrical distribution)
        "ATKR",   # Atkore Inc (electrical raceway)
        
        # Construction Materials
        "VMC",    # Vulcan Materials (aggregates - construction proxy)
        "MLM",    # Martin Marietta Materials
    ],
    
    # === GLOBAL MINING COMPANIES (Emerging Markets Exposure) ===
    "Emerging Market Miners": [
        "CHGCY",  # China Gold International (China)
        "MMTLF",  # MMG Limited (China - owns Las Bambas Peru)
        "JCHXF",  # Jiangxi Copper (China)
        "NGLOY",  # Anglo American (already listed)
    ],
    
    # === EXPLORATION & DEVELOPMENT (High Beta) ===
    "Junior Miners": [
        "COPRF",  # Copper Mountain Mining
        "ARIZF",  # Arizona Metals Corp
        "CBDRF",  # Canterra Minerals
        "CHALF",  # Chakana Copper Corp
        "NFGC",   # New Found Gold (gold but check correlation)
    ],
}

# Flatten the universe
ALL_TICKERS = [BENCHMARK]
for category, tickers in UNIVERSE.items():
    ALL_TICKERS.extend(tickers)

# Remove duplicates
ALL_TICKERS = list(set(ALL_TICKERS))

# Analysis Parameters
PERIOD = "5y"  # Use 5 years for more recent correlation
MIN_CORRELATION = 0.50  # Minimum correlation to copper (adjustable)
MIN_DATA_POINTS = 750   # Minimum trading days required (3 years ≈ 750 days)
TOP_N_PAIRS = 30        # Number of top pairs to display

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def download_with_retry(tickers, period, max_retries=3):
    """Download data with retry logic for failed tickers."""
    for attempt in range(max_retries):
        try:
            print(f"Downloading data (attempt {attempt + 1}/{max_retries})...")
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if 'Close' in data.columns:
                return data['Close']
            else:
                return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("All attempts failed. Trying individual downloads...")
                return download_individually(tickers, period)
    return pd.DataFrame()

def download_individually(tickers, period):
    """Download tickers one by one (slower but more reliable)."""
    data_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)['Close']
            if not df.empty and len(df) > MIN_DATA_POINTS:
                data_dict[ticker] = df
                print(f"✓ {ticker}")
        except:
            print(f"✗ {ticker} (failed)")
    
    if data_dict:
        return pd.DataFrame(data_dict)
    return pd.DataFrame()

def calculate_correlation_metrics(benchmark_series, candidate_series):
    """Calculate correlation and cointegration metrics."""
    # Align the series
    combined = pd.concat([benchmark_series, candidate_series], axis=1).dropna()
    
    if len(combined) < MIN_DATA_POINTS:
        return None
    
    bench = combined.iloc[:, 0]
    cand = combined.iloc[:, 1]
    
    # Pearson Correlation
    corr, p_value = pearsonr(bench, cand)
    
    # Log transformation for cointegration test
    log_bench = np.log(bench)
    log_cand = np.log(cand)
    
    # Simple linear regression for spread
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(log_cand, log_bench)
    spread = log_bench - (slope * log_cand + intercept)
    
    # ADF test on spread (cointegration)
    try:
        adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread, maxlag=20)
        cointegrated = adf_pvalue < 0.05
    except:
        adf_pvalue = 1.0
        cointegrated = False
    
    # Calculate beta (hedge ratio)
    returns_bench = bench.pct_change().dropna()
    returns_cand = cand.pct_change().dropna()
    aligned_rets = pd.concat([returns_bench, returns_cand], axis=1).dropna()
    
    if len(aligned_rets) > 50:
        cov_matrix = aligned_rets.cov()
        beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1] if cov_matrix.iloc[1, 1] != 0 else 0
    else:
        beta = slope
    
    # Volatility
    volatility = cand.pct_change().std() * np.sqrt(252) * 100  # Annualized %
    
    return {
        'correlation': corr,
        'corr_pvalue': p_value,
        'cointegrated': cointegrated,
        'adf_pvalue': adf_pvalue,
        'beta': beta,
        'volatility': volatility,
        'data_points': len(combined),
        'r_squared': r_value**2
    }

# =============================================================================
# DATA DOWNLOAD & PROCESSING
# =============================================================================

print(f"{'='*70}")
print(f"COPPER PAIRS TRADING - UNIVERSE SCANNER")
print(f"{'='*70}")
print(f"Total tickers to scan: {len(ALL_TICKERS)}")
print(f"Benchmark: {BENCHMARK}")
print(f"Period: {PERIOD}")
print(f"{'='*70}\n")

# Download data
raw_data = download_with_retry(ALL_TICKERS, PERIOD)

if raw_data.empty or BENCHMARK not in raw_data.columns:
    print("ERROR: Failed to download benchmark (HG=F). Exiting.")
    exit(1)

print(f"\nSuccessfully downloaded {len(raw_data.columns)} securities.")

# Clean data
df_clean = raw_data.ffill().dropna(axis=1, thresh=MIN_DATA_POINTS)
print(f"After filtering: {len(df_clean.columns)} securities with sufficient data.\n")

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

print("Analyzing correlations to copper futures...")
benchmark_series = df_clean[BENCHMARK]
results = []

for ticker in df_clean.columns:
    if ticker == BENCHMARK:
        continue
    
    candidate_series = df_clean[ticker]
    metrics = calculate_correlation_metrics(benchmark_series, candidate_series)
    
    if metrics and metrics['correlation'] >= MIN_CORRELATION:
        # Find category
        category = "Unknown"
        for cat, tickers in UNIVERSE.items():
            if ticker in tickers:
                category = cat
                break
        
        results.append({
            'Ticker': ticker,
            'Category': category,
            'Correlation': metrics['correlation'],
            'P-Value': metrics['corr_pvalue'],
            'Cointegrated': 'YES' if metrics['cointegrated'] else 'NO',
            'ADF P-Value': metrics['adf_pvalue'],
            'Beta': metrics['beta'],
            'R²': metrics['r_squared'],
            'Volatility (%)': metrics['volatility'],
            'Data Points': metrics['data_points']
        })

# Sort by correlation (descending)
results_df = pd.DataFrame(results).sort_values('Correlation', ascending=False)

print(f"\n{'='*70}")
print(f"TOP {TOP_N_PAIRS} PAIRS MOST CORRELATED TO COPPER")
print(f"{'='*70}\n")
print(results_df.head(TOP_N_PAIRS).to_string(index=False))

# =============================================================================
# EXPORT RESULTS
# =============================================================================

# Save analysis to CSV
output_file = "copper_pairs_analysis.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✓ Full results saved to: {output_file}")

# =============================================================================
# EXPORT NORMALIZED PRICE DATA (BASE 100)
# =============================================================================

print(f"\nExporting normalized price data...")

# Select top N tickers
n_tickers = min(TOP_N_PAIRS, len(results_df))
top_tickers = results_df.head(n_tickers)['Ticker'].tolist()

# Add benchmark
tickers_for_export = [BENCHMARK] + top_tickers

# Filter dataframe to only include selected tickers
df_export = df_clean[[t for t in tickers_for_export if t in df_clean.columns]].copy()

# Normalize to Base 100 (first day = 100)
df_normalized = (df_export / df_export.iloc[0]) * 100

# Reset index to make Date a column
df_normalized.reset_index(inplace=True)
df_normalized.rename(columns={'Date': 'Date'}, inplace=True)

# Sort columns: Date first, then alphabetically
cols = ['Date'] + sorted([c for c in df_normalized.columns if c != 'Date'])
df_normalized = df_normalized[cols]

# Save to CSV
normalized_file = f"copper_pairs_normalized_top{n_tickers}.csv"
df_normalized.to_csv(normalized_file, index=False)

print(f"✓ Normalized data saved to: {normalized_file}")
print(f"  - Columns: {len(df_normalized.columns)} (Date + {len(df_normalized.columns)-1} tickers)")
print(f"  - Rows: {len(df_normalized)} days")
print(f"  - Base: 100 (first day = 100.0)")

# Create recommended ticker list for your trading script
top_tickers_list = [BENCHMARK] + results_df.head(TOP_N_PAIRS)['Ticker'].tolist()

print(f"\n{'='*70}")
print(f"RECOMMENDED TICKER LIST FOR TRADING SCRIPT")
print(f"{'='*70}")
print(f"TICKERS = {top_tickers_list}")
print(f"{'='*70}\n")

# =============================================================================
# ENHANCED VISUALIZATION
# =============================================================================

print("Generating visualization...")

# Normalize to Base 100
df_rebase = (df_clean / df_clean.iloc[0]) * 100

# Select top pairs for plotting
top_ticker_plot_list = results_df.head(15)['Ticker'].tolist()
tickers_to_plot = [BENCHMARK] + top_ticker_plot_list

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    subplot_titles=("Price Performance (Base 100)", "Correlation Matrix (Top 15)"),
    vertical_spacing=0.15
)

# === SUBPLOT 1: Price Performance ===
for ticker in tickers_to_plot:
    if ticker not in df_rebase.columns:
        continue
    
    if ticker == BENCHMARK:
        # Copper Futures - Prominent
        line_color = '#B22222'
        line_width = 4
        line_dash = 'solid'
        opacity = 1.0
        visible = True
        name = f" {ticker} (Copper Benchmark)"
    else:
        # Find if cointegrated
        row = results_df[results_df['Ticker'] == ticker]
        is_cointegrated = row['Cointegrated'].values[0] == 'YES' if not row.empty else False
        corr_val = row['Correlation'].values[0] if not row.empty else 0
        
        if is_cointegrated:
            line_color = '#228B22'  # Green for cointegrated
            name_prefix = "✓"
        else:
            line_color = '#4169E1'  # Blue for high correlation only
            name_prefix = "○"
        
        line_width = 2
        line_dash = 'solid'
        opacity = 0.7
        visible = "legendonly"
        name = f"{name_prefix} {ticker} (ρ={corr_val:.3f})"
    
    fig.add_trace(
        go.Scatter(
            x=df_rebase.index,
            y=df_rebase[ticker],
            mode='lines',
            name=name,
            line=dict(color=line_color, width=line_width, dash=line_dash),
            opacity=opacity,
            visible=visible,
            hovertemplate=f'{ticker}: %{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

# === SUBPLOT 2: Correlation Heatmap ===
top_15 = [BENCHMARK] + results_df.head(14)['Ticker'].tolist()
top_15_data = df_clean[[t for t in top_15 if t in df_clean.columns]]
corr_matrix = top_15_data.corr()

fig.add_trace(
    go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation", x=1.15)
    ),
    row=2, col=1
)

# === LAYOUT ===
fig.update_xaxes(
    title_text="Date",
    showgrid=True,
    gridcolor='LightGrey',
    row=1, col=1
)

fig.update_yaxes(
    title_text="Performance (Base 100)",
    type="log",
    showgrid=True,
    gridcolor='LightGrey',
    row=1, col=1
)

fig.update_xaxes(
    tickangle=-45,
    row=2, col=1
)

fig.update_yaxes(
    tickangle=0,
    row=2, col=1
)

fig.update_layout(
    title=dict(
        text=f"Copper Pairs Trading - Top {len(top_ticker_plot_list)} Correlated Securities ({PERIOD})",
        font=dict(size=18, family="Arial", color="#2c3e50")
    ),
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05,
        font=dict(size=10)
    ),
    height=1000,
    showlegend=True
)

# Add range selector to first subplot
fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(step="all", label="MAX")
        ])
    ),
    row=1, col=1
)

fig.show()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print(f"\n{'='*70}")
print(f"SUMMARY STATISTICS")
print(f"{'='*70}")
print(f"Total securities analyzed: {len(df_clean.columns) - 1}")
print(f"Securities with correlation > {MIN_CORRELATION}: {len(results_df)}")
print(f"Cointegrated pairs found: {results_df['Cointegrated'].value_counts().get('YES', 0)}")
print(f"Average correlation (top 30): {results_df.head(30)['Correlation'].mean():.3f}")
print(f"Highest correlation: {results_df.iloc[0]['Ticker']} ({results_df.iloc[0]['Correlation']:.3f})")
print(f"{'='*70}\n")

# Category breakdown
print("CATEGORY BREAKDOWN (Top 30):")
category_counts = results_df.head(30)['Category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category}: {count}")

print(f"\n{'='*70}")
print("✓ Analysis complete!")
print(f"{'='*70}\n")
