import sys
import os
import json
import warnings
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

# GUI Framework (PyQt6)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QFileDialog, QProgressBar, 
    QGroupBox, QFormLayout, QMessageBox, QSplitter, QDockWidget,
    QTabWidget, QInputDialog, QDialog, QDialogButtonBox, QLineEdit, QComboBox,
    QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QAction, QColor, QIcon

# Data Science & Math
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from itertools import product

# Visualization (Matplotlib)
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Configuration
warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('QtAgg')

# --- STYLE: MODERN DARK THEME ---
DARK_THEME_QSS = """
QMainWindow { background-color: #1e1e1e; }
QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 10pt; }
QDockWidget { titlebar-close-icon: url(close.png); titlebar-normal-icon: url(float.png); }
QDockWidget::title { background: #2d2d2d; padding-left: 5px; padding-top: 4px; }

/* Buttons */
QPushButton {
    background-color: #0d6efd; border: none; border-radius: 4px;
    color: white; padding: 6px 12px; font-weight: bold;
}
QPushButton:hover { background-color: #0b5ed7; }
QPushButton:pressed { background-color: #0a58ca; }
QPushButton:disabled { background-color: #444; color: #888; }

/* Inputs */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #2d2d2d; border: 1px solid #444; border-radius: 3px;
    color: white; padding: 3px;
}

/* Tables */
QTableWidget {
    background-color: #252526; gridline-color: #3e3e42; border: none;
    selection-background-color: #264f78;
}
QHeaderView::section {
    background-color: #333337; color: #cccccc; border: 1px solid #3e3e42; padding: 4px;
}

/* Tabs */
QTabWidget::pane { border: 1px solid #3e3e42; }
QTabBar::tab {
    background: #2d2d2d; color: #888; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px;
}
QTabBar::tab:selected { background: #1e1e1e; color: white; border-top: 2px solid #0d6efd; }
"""

# --- DATA DOWNLOAD DIALOG ---
class YFinanceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Market Data")
        self.setFixedWidth(400)
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Tickers (comma separated):"))
        self.txt_tickers = QLineEdit("HG=F, GLD, GDX, SLV, SIL, COPX, FCX, SCCO, BHP, RIO")
        layout.addWidget(self.txt_tickers)
        
        layout.addWidget(QLabel("Start Date (YYYY-MM-DD):"))
        self.txt_start = QLineEdit((datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'))
        layout.addWidget(self.txt_start)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_data(self):
        tickers = [t.strip() for t in self.txt_tickers.text().split(',')]
        start = self.txt_start.text()
        return tickers, start

# --- CORE TRADING ENGINE (FIXED) ---
class TradingStrategy:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.assets: List[str] = []
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def fetch_yfinance(self, tickers: List[str], start_date: str):
        """Downloads historical data using YFinance."""
        print(f"Downloading {tickers} from {start_date}...")
        try:
            data = yf.download(tickers, start=start_date, progress=False)['Close']
            if data.empty: 
                raise ValueError("No data returned from Yahoo Finance.")
            self.df = data.dropna()
            self.assets = self.df.columns.tolist()
        except Exception as e:
            raise ValueError(f"Data Download Error: {e}")

    def load_csv(self, filepath: str):
        """Loads data from a local CSV file."""
        try:
            self.df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.df = self.df.apply(pd.to_numeric, errors='coerce').dropna()
            self.assets = self.df.columns.tolist()
        except Exception as e:
            raise ValueError(f"CSV Load Error: {e}")

    def split_data(self, train_pct: float = 0.2):
        """
        Split data into train/test to avoid look-ahead bias.
        Cointegration will be tested ONLY on training data.
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded. Please load data first.")
        
        split_idx = int(len(self.df) * train_pct)
        self.train_data = self.df.iloc[:split_idx].copy()
        self.test_data = self.df.iloc[split_idx:].copy()
        
        print(f"Train: {self.train_data.index[0]} to {self.train_data.index[-1]} ({len(self.train_data)} days)")
        print(f"Test:  {self.test_data.index[0]} to {self.test_data.index[-1]} ({len(self.test_data)} days)")

    def find_cointegrated_pairs(self, data: pd.DataFrame, corr_threshold: float = 0.85, pvalue_threshold: float = 0.05):
        """
        Scan de cointégration robuste.
        """
        numeric_df = data.select_dtypes(include=[np.number])
        log_df = np.log(numeric_df).replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        
        corr = numeric_df.corr()
        pairs = []
        assets = log_df.columns.tolist()
        n = len(assets)
        
        for i in range(n):
            for j in range(i + 1, n):
                s1, s2 = assets[i], assets[j]
                
                # correlation filter
                if corr.loc[s1, s2] < corr_threshold:
                    continue
                
                try:
                    # linear regression
                    res = linregress(log_df[s2], log_df[s1])
                    spread = log_df[s1] - (res.slope * log_df[s2] + res.intercept)
                    
                  
                    adf_stats = adfuller(spread.dropna())
                    pvalue = adf_stats[1] 
                    
                    if pvalue < pvalue_threshold:
                        pairs.append((s1, s2))
                except Exception:
                    continue
        
        return pairs
    def run_monte_carlo(self, returns_series: pd.Series, simulations: int = 50):
        """
        Monte Carlo now properly aligned and uses only realized returns.
        No future data leakage.
        """
        sim_results = []
        daily_rets = returns_series.dropna().values
        
        if len(daily_rets) == 0:
            return []
        
        horizon = len(daily_rets)

        for _ in range(simulations):
            # Bootstrap resampling with replacement
            shuffled_rets = np.random.choice(daily_rets, size=horizon, replace=True)
            
            # Reconstruct equity curve (base 1.0)
            cum_ret_sim = np.cumprod(1 + shuffled_rets)
            sim_results.append(cum_ret_sim)
        
        return sim_results

    def backtest_pair(self, s1, s2, data: pd.DataFrame, params: dict):
        """
        Improved position management and cost calculation.
        Properly tracks actual trades (not just position changes).
        Implements cooldown period after stop loss hits.
        """
        # 1. Prepare Data
        log_df = np.log(data[[s1, s2]])
        y, x = log_df[s1], sm.add_constant(log_df[s2])

        # 2. Rolling OLS (Hedge Ratio) with proper lag
        # Add 2-day lag to ensure no look-ahead
        model = RollingOLS(y, x, window=params['beta_window'])
        rolling_res = model.fit()
        res_params = rolling_res.params.shift(2)  # 2-day lag
        beta = res_params[s2].fillna(method='ffill')
        alpha = res_params['const'].fillna(method='ffill')

        # 3. Z-Score Calculation
        spread = y - (beta * log_df[s2] + alpha)
        roll_mean = spread.rolling(window=params['z_window']).mean()
        roll_std = spread.rolling(window=params['z_window']).std()
        
        # Avoid division by zero
        roll_std = roll_std.replace(0, np.nan)
        z_score = (spread - roll_mean) / roll_std

        # 4. Generate Entry Signals
        entry_long = z_score < -params['entry_z']
        entry_short = z_score > params['entry_z']
        
        # 5. FIX: Improved Position Management with Cooldown
        positions = pd.Series(0, index=z_score.index)
        current_pos = 0
        cooldown_end_idx = -1  # Index until which trading is disabled
        stop_loss_count = 0  # Track number of stop losses hit
        
        enable_cooldown = params.get('enable_cooldown', False)
        cooldown_days = params.get('cooldown_days', 20)
        
        for i in range(len(z_score)):
            if pd.isna(z_score.iloc[i]):
                positions.iloc[i] = current_pos
                continue
            
            z = z_score.iloc[i]
            
            # Check if we're still in cooldown period
            if enable_cooldown and i <= cooldown_end_idx:
                # Force close any existing position and prevent new trades
                current_pos = 0
                positions.iloc[i] = 0
                continue
            
            # Entry logic (only if not in position and not in cooldown)
            if current_pos == 0:
                if entry_long.iloc[i]:
                    current_pos = 1
                elif entry_short.iloc[i]:
                    current_pos = -1
            
            # Exit logic (Take Profit)
            elif current_pos == 1 and z > -params['exit_z']:
                current_pos = 0
            elif current_pos == -1 and z < params['exit_z']:
                current_pos = 0
            
            # Stop Loss with Cooldown Trigger
            if abs(z) > params['stop_loss_z'] and current_pos != 0:
                current_pos = 0  # Close position immediately
                stop_loss_count += 1
                
                # Activate cooldown period
                if enable_cooldown:
                    cooldown_end_idx = i + cooldown_days
                    # Note: cooldown_end_idx might exceed array length, that's OK
            
            positions.iloc[i] = current_pos

        # 6. Calculate PnL
        pos_delayed = positions.shift(1).fillna(0)
        ret1 = data[s1].pct_change()
        ret2 = data[s2].pct_change()
        
        # Hedge ratio for returns
        beta_aligned = beta.reindex(ret1.index).fillna(method='ffill')
        spread_ret = ret1 - beta_aligned * ret2
        
        pnl = pos_delayed * spread_ret
        
        # 7. Cost Calculation
        # Count actual trades (position changes from 0 to ±1 or vice versa)
        pos_changes = positions.diff().fillna(0)
        actual_trades = (pos_changes != 0).astype(int)
        total_trades = actual_trades.sum()
        
        # Apply costs when position changes (entry OR exit)
        total_cost = params['tx_cost'] + params['slippage']
        
        # Costs are applied whenever there's a trade (position change)
        # We use the CURRENT position size (not delayed) to calculate the notional traded
        costs = actual_trades * total_cost
        
        net_pnl = pnl - costs
        
        return net_pnl.fillna(0), z_score, positions, total_trades, stop_loss_count

    def backtest_portfolio(self, pairs: List[Tuple[str, str]], data: pd.DataFrame, params: dict):
        """
        FIX #5: Portfolio construction with proper risk parity.
        NEW: Tracks stop loss events per pair.
        """
        all_pnls = pd.DataFrame(index=data.index)
        pair_data = {}
        total_portfolio_trades = 0
        total_stop_losses = 0
        
        for s1, s2 in pairs:
            pnl, z, pos, num_trades, sl_count = self.backtest_pair(s1, s2, data, params)
            
            if pnl.std() != 0:
                sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252)
                name = f"{s1}-{s2}"
                all_pnls[name] = pnl
                pair_data[name] = {
                    'sharpe': sharpe, 
                    'z': z, 
                    'pos': pos, 
                    'pnl': pnl, 
                    'trades': num_trades,
                    'stop_losses': sl_count
                }
                total_portfolio_trades += num_trades
                total_stop_losses += sl_count

        if all_pnls.empty:
            return None, None, None, 0, 0

        final_pnls = all_pnls.fillna(0)
        
        if params.get('risk_parity', True):
            # Use configurable volatility window
            vol_window = params.get('risk_parity_window', 60)
            inv_vol = 1.0 / final_pnls.rolling(vol_window).std().shift(1).replace(0, np.nan)
            weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
            
            # Cap maximum weight to avoid concentration
            max_weight = params.get('max_pair_weight', 0.3)
            weights = weights.clip(upper=max_weight)
            weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
            
            port_ret = (final_pnls * weights).sum(axis=1)
        else:
            port_ret = final_pnls.mean(axis=1)

        cum_ret = (1 + port_ret).cumprod()
        
        return cum_ret, pair_data, port_ret, total_portfolio_trades, total_stop_losses

# --- WORKER THREAD (BACKGROUND TASKS) ---
class BacktestWorker(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(object, object, object, object, object, object)
    error_signal = pyqtSignal(str)

    def __init__(self, strategy, params, use_walk_forward):
        super().__init__()
        self.strategy = strategy
        self.params = params
        self.use_walk_forward = use_walk_forward

    def run(self):
        try:
            self.strategy.split_data(train_pct=0.2)
            
            self.progress_signal.emit(10)
            pairs = self.strategy.find_cointegrated_pairs(
                self.strategy.train_data,
                corr_threshold=0.85,
                pvalue_threshold=0.05
            )

            # 1. Cointegration Scan (against HG=F)
            target = "HG=F"
            if target not in self.strategy.assets:
                self.error_signal.emit(f"Le ticker {target} est absent des données.")
                return

            log_df = np.log(self.strategy.df)
            corr = self.strategy.df.corr()

            # compute correlation with HG=F (excluding HG=F itself)
            correlations_with_target = corr[target].drop(labels=[target]).sort_values(ascending=False)

            # take the n actifs the most correlated
            n = 8
            top_5_tickers = correlations_with_target.head(5).index.tolist()
            top_10_tickers = correlations_with_target.head(10).index.tolist()
            top_n_tickers = correlations_with_target.head(n).index.tolist()
            pairs = []
            for s2 in top_n_tickers:
                s1 = target # HG=F
                res = linregress(log_df[s2], log_df[s1])
                spread = log_df[s1] - (res.slope * log_df[s2] + res.intercept)
                
                # Test ADF
                if adfuller(spread)[1] < 0.05: 
                    pairs.append((s1, s2))
                else:
                    pairs.append((s1, s2)) 
                    print(f"Paire {s1}-{s2} écartée : Non cointégrée (p-val > 0.05)")
            if not pairs:
                self.error_signal.emit("No cointegrated pairs found in training data. Try different assets or adjust thresholds.")
                return
            
            print(f"Found {len(pairs)} cointegrated pairs: {pairs}")
            
            if self.use_walk_forward:
                # Walk-Forward Analysis
                self.progress_signal.emit(30)
                train_result = self.strategy.backtest_portfolio(
                    pairs, self.strategy.train_data, self.params
                )
                
                self.progress_signal.emit(60)
                test_result = self.strategy.backtest_portfolio(
                    pairs, self.strategy.test_data, self.params
                )
                
                if train_result[0] is None or test_result[0] is None:
                    self.error_signal.emit("Backtest failed on train or test data.")
                    return
                
                # Combine results
                cum_ret_train, pair_data_train, port_ret_train, trades_train, sl_train = train_result
                cum_ret_test, pair_data_test, port_ret_test, trades_test, sl_test = test_result
                
                # Test period should continue from where train ended, not restart at 1.0
                train_final_value = cum_ret_train.iloc[-1]
                
                # Scale test returns to continue from train's final value
                # Convert test cum_ret back to simple returns, then rebuild from train endpoint
                test_simple_returns = cum_ret_test.pct_change().fillna(0)
                cum_ret_test_chained = train_final_value * (1 + test_simple_returns).cumprod()
                
                # Now concatenate
                cum_ret = pd.concat([cum_ret_train, cum_ret_test_chained])
                port_ret = pd.concat([port_ret_train, port_ret_test])
                
                # Merge pair data (use test performance for metrics)
                pair_data = pair_data_test
                total_trades = trades_train + trades_test
                total_stop_losses = sl_train + sl_test
                
                # Mark split point
                split_date = self.strategy.test_data.index[0]
                
            else:
                # Standard backtest on full data
                self.progress_signal.emit(50)
                result = self.strategy.backtest_portfolio(
                    pairs, self.strategy.df, self.params
                )
                
                if result[0] is None:
                    self.error_signal.emit("No valid pairs found. Try relaxing filters.")
                    return
                
                cum_ret, pair_data, port_ret, total_trades, total_stop_losses = result
                split_date = None
            
            self.progress_signal.emit(90)
            
            # Monte Carlo simulations
            mc_sims = self.strategy.run_monte_carlo(port_ret, simulations=50)
            
            # Calculate metrics
            total_return = cum_ret.iloc[-1] - 1.0
            
            if total_trades > 0:
                # Calculate total costs paid across all pairs
                total_costs_paid = total_trades * (self.params.get('tx_cost', 0) + self.params.get('slippage', 0))
                
                avg_cost_per_trade = total_costs_paid / total_trades
                avg_trade_ret = total_return / total_trades
            else:
                avg_cost_per_trade = 0
                avg_trade_ret = 0
            
            extra_metrics = {
                'total_trades': total_trades,
                'avg_trade_ret': avg_trade_ret,
                'avg_cost_per_trade': avg_cost_per_trade,
                'split_date': split_date,
                'total_stop_losses': total_stop_losses
            }
            
            self.progress_signal.emit(100)
            self.result_signal.emit(cum_ret, pair_data, port_ret, mc_sims, extra_metrics, pairs)

        except Exception as e:
            import traceback
            self.error_signal.emit(f"{str(e)}\n\n{traceback.format_exc()}")

class OptimizerWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict, float, dict)
    finished_signal = pyqtSignal()

    def __init__(self, strategy, grid, metric_type, fixed_costs):
        super().__init__()
        self.strategy = strategy
        self.grid = grid
        self.metric_type = metric_type
        self.costs = fixed_costs

    def calculate_metric(self, port_ret):
        """Helper to calculate specific metrics from portfolio returns."""
        if port_ret.empty or port_ret.std() == 0:
            return -999.0
        
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        
        if self.metric_type == "Sharpe Ratio":
            return ann_ret / ann_vol if ann_vol != 0 else -999.0
            
        elif self.metric_type == "Sortino Ratio":
            downside = port_ret[port_ret < 0].std() * np.sqrt(252)
            return ann_ret / downside if downside != 0 else -999.0
            
        elif self.metric_type == "Total Return":
            return (1 + port_ret).cumprod().iloc[-1] - 1
            
        elif self.metric_type == "Minimize Max Drawdown":
            cum_ret = (1 + port_ret).cumprod()
            dd = (cum_ret / cum_ret.cummax()) - 1
            return -dd.min()  # we want to minimize
            
        return -999.0

    def run(self):
        # Optimize only on training data
        self.strategy.split_data(train_pct=0.2)
        
        # Find pairs on training data
        pairs = self.strategy.find_cointegrated_pairs(
            self.strategy.train_data,
            corr_threshold=0.85,
            pvalue_threshold=0.05
        )
        
        if not pairs:
            self.log_signal.emit("No cointegrated pairs found for optimization.")
            self.result_signal.emit({}, -999.0, {})
            self.finished_signal.emit()
            return
        
        # Generate combinations
        keys = list(self.grid.keys())
        combinations = list(product(*self.grid.values()))
        total_combos = len(combinations)
        
        best_score = -float('inf')
        best_params = {}
        all_results = {}

        for i, values in enumerate(combinations):
            current_params = dict(zip(keys, values))
            current_params.update(self.costs)
            current_params['risk_parity'] = True
            current_params['risk_parity_window'] = 60
            current_params['max_pair_weight'] = 0.3

            try:
                # Backtest on TRAINING data only
                result = self.strategy.backtest_portfolio(
                    pairs, self.strategy.train_data, current_params
                )
                
                if result[0] is None:
                    score = -999.0
                else:
                    _, _, port_ret, _, _ = result  # Unpack with stop losses
                    score = self.calculate_metric(port_ret)
                
                all_results[str(current_params)] = score

                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()
                    self.log_signal.emit(f"New Best: {score:.4f} | Params: {current_params}")

            except Exception as e:
                pass

            self.progress_signal.emit(int((i + 1) / total_combos * 100))
        
        self.result_signal.emit(best_params, best_score, all_results)
        self.finished_signal.emit()

# --- MAIN APPLICATION WINDOW ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statistical Arbitrage App (Fixed)")
        self.resize(1600, 900)
        self.setStyleSheet(DARK_THEME_QSS)
        
        self.strategy = TradingStrategy()
        self.pair_data_cache = {} 
        self.cointegrated_pairs = []
        self.init_ui()
        
        self.settings = QSettings("Arbitrage Strategies Backtesting", "ben-245dev-fixed")
        self.load_settings()

    def init_ui(self):
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowNestedDocks)

        # --- DOCK: CONTROLS ---
        self.dock_controls = QDockWidget("Strategy Control Center", self)
        self.dock_controls.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.control_widget = QWidget()
        self.layout_controls = QFormLayout()
        
        # Data Section
        btn_layout = QHBoxLayout()
        self.btn_load_csv = QPushButton("Load CSV")
        self.btn_load_yf = QPushButton("Yahoo Finance")
        self.btn_load_csv.clicked.connect(self.load_csv)
        self.btn_load_yf.clicked.connect(self.load_yf)
        btn_layout.addWidget(self.btn_load_csv)
        btn_layout.addWidget(self.btn_load_yf)
        self.layout_controls.addRow("Data Source:", btn_layout)
        
        # Walk-Forward Option
        self.chk_walk_forward = QCheckBox("Use Walk-Forward Analysis (Train/Test Split)")
        self.chk_walk_forward.setChecked(True)
        self.layout_controls.addRow(self.chk_walk_forward)
        
        self.layout_controls.addRow(QLabel("--- Rolling Window Parameters ---"))
        self.spin_beta = QSpinBox()
        self.spin_beta.setRange(1, 100000)
        self.spin_beta.setValue(84)
        self.layout_controls.addRow("Beta Window (Days):", self.spin_beta)
        
        self.spin_z = QSpinBox()
        self.spin_z.setRange(1, 100000)
        self.spin_z.setValue(21)
        self.layout_controls.addRow("Z-Score Window (Days):", self.spin_z)
        
        self.layout_controls.addRow(QLabel("--- Trading Logic (Sigma) ---"))
        self.spin_entry = QDoubleSpinBox()
        self.spin_entry.setValue(1.5)
        self.spin_entry.setRange(-100000, 100000)
        self.spin_entry.setSingleStep(0.1)
        self.layout_controls.addRow("Entry Threshold:", self.spin_entry)
        
        self.spin_exit = QDoubleSpinBox()
        self.spin_exit.setValue(-1.5)
        self.spin_exit.setRange(-100000, 100000)
        self.spin_exit.setSingleStep(0.1)
        self.layout_controls.addRow("Exit Threshold (TP):", self.spin_exit)
        
        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setValue(4.5)
        self.spin_stop.setRange(-100000, 100000)
        self.spin_stop.setSingleStep(0.1)
        self.layout_controls.addRow("Stop Loss:", self.spin_stop)
        
        # Cooldown after stop loss
        self.layout_controls.addRow(QLabel("--- Stop Loss Cooldown ---"))
        
        self.chk_enable_cooldown = QCheckBox("Enable Cooldown After Stop Loss")
        self.chk_enable_cooldown.setChecked(False)
        self.chk_enable_cooldown.setToolTip("Temporarily disable trading on a pair after stop loss is hit")
        self.layout_controls.addRow(self.chk_enable_cooldown)
        
        self.spin_cooldown_days = QSpinBox()
        self.spin_cooldown_days.setRange(1, 1000)
        self.spin_cooldown_days.setValue(20)
        self.spin_cooldown_days.setToolTip("Number of days to stop trading this pair after stop loss")
        self.layout_controls.addRow("Cooldown Duration (Days):", self.spin_cooldown_days)
        
        self.layout_controls.addRow(QLabel("--- Execution Costs ---"))
        self.spin_cost = QDoubleSpinBox()
        self.spin_cost.setValue(0.001)
        self.spin_cost.setRange(0, 1)
        self.spin_cost.setDecimals(4)
        self.layout_controls.addRow("Commission (%):", self.spin_cost)
        
        self.spin_slip = QDoubleSpinBox()
        self.spin_slip.setValue(0.0005)
        self.spin_slip.setRange(0, 1)
        self.spin_slip.setDecimals(4)
        self.layout_controls.addRow("Slippage (%):", self.spin_slip)
        
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setStyleSheet("background-color: #198754; font-size: 14px; height: 40px;")
        self.btn_run.clicked.connect(self.run_backtest)
        self.layout_controls.addRow(self.btn_run)
        
        self.layout_controls.addRow(QLabel("--- Optimization Settings ---"))
        
        self.combo_metric = QComboBox()
        self.combo_metric.addItems(["Sharpe Ratio", "Sortino Ratio", "Total Return", "Minimize Max Drawdown"])
        self.layout_controls.addRow("Target Metric:", self.combo_metric)
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Quick (Coarse Grid)", "Exhaustive (Fine Grid)"])
        self.layout_controls.addRow("Search Mode:", self.combo_mode)
        
        self.btn_optimize = QPushButton("FIND BEST PARAMS")
        self.btn_optimize.setStyleSheet("background-color: #d63384; font-weight: bold; height: 40px;")
        self.btn_optimize.clicked.connect(self.run_optimization)
        self.layout_controls.addRow(self.btn_optimize)

        save_load_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save Config")
        self.btn_save.clicked.connect(self.save_config)
        self.btn_load_cfg = QPushButton("Load Config")
        self.btn_load_cfg.clicked.connect(self.load_config)
        save_load_layout.addWidget(self.btn_save)
        save_load_layout.addWidget(self.btn_load_cfg)
        self.layout_controls.addRow(save_load_layout)

        self.progress = QProgressBar()
        self.layout_controls.addRow(self.progress)
        
        self.control_widget.setLayout(self.layout_controls)
        self.dock_controls.setWidget(self.control_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_controls)

        # --- DOCK: RESULTS TABLE ---
        self.dock_table = QDockWidget("Performance Attribution", self)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Pair", "Sharpe", "Total Return", "Stop Losses", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemClicked.connect(self.drill_down_pair)
        self.dock_table.setWidget(self.table)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_table)

        # --- DOCK: METRICS ---
        self.dock_metrics = QDockWidget("Financial Tear Sheet", self)
        self.metrics_widget = QWidget()
        self.metrics_layout = QVBoxLayout()
        self.lbl_sharpe = QLabel("Sharpe: N/A")
        self.lbl_sortino = QLabel("Sortino: N/A")
        self.lbl_cagr = QLabel("CAGR: N/A")
        self.lbl_maxdd = QLabel("Max DD: N/A")
        self.lbl_trades = QLabel("Nb Trades: N/A")
        self.lbl_avg_ret = QLabel("Avg Ret/Trade: N/A")
        self.lbl_avg_cost = QLabel("Avg Cost/Trade: N/A")
        self.lbl_pairs_found = QLabel("Pairs Found: N/A")
        self.lbl_stop_losses = QLabel("Stop Losses Hit: N/A")
        self.metrics_layout.addWidget(self.lbl_sharpe)
        self.metrics_layout.addWidget(self.lbl_sortino)
        self.metrics_layout.addWidget(self.lbl_cagr)
        self.metrics_layout.addWidget(self.lbl_maxdd)
        self.metrics_layout.addWidget(self.lbl_trades)
        self.metrics_layout.addWidget(self.lbl_avg_ret)
        self.metrics_layout.addWidget(self.lbl_avg_cost)
        self.metrics_layout.addWidget(self.lbl_pairs_found)
        self.metrics_layout.addWidget(self.lbl_stop_losses)
        self.metrics_layout.addStretch()
        self.metrics_widget.setLayout(self.metrics_layout)
        self.dock_metrics.setWidget(self.metrics_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_metrics)

        # --- CENTRAL: CHARTS ---
        self.tab_charts = QTabWidget()
        self.fig_equity = Figure(figsize=(8, 5), facecolor='#1e1e1e')
        self.canvas_equity = FigureCanvas(self.fig_equity)
        self.tab_charts.addTab(self.canvas_equity, "Portfolio Equity")
        
        self.fig_detail = Figure(figsize=(8, 5), facecolor='#1e1e1e')
        self.canvas_detail = FigureCanvas(self.fig_detail)
        self.tab_charts.addTab(self.canvas_detail, "Pair Drill-Down")
        
        self.setCentralWidget(self.tab_charts)

    def reset_ui(self):
        """Clears all charts and tables."""
        self.table.setRowCount(0)
        
        self.fig_equity.clear()
        ax = self.fig_equity.add_subplot(111)
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, "Running Simulation...", color='white', ha='center')
        ax.axis('off')
        self.canvas_equity.draw()
        
        self.fig_detail.clear()
        self.canvas_detail.draw()
        
        self.lbl_sharpe.setText("Sharpe: ...")
        self.lbl_sortino.setText("Sortino: ...")
        self.lbl_cagr.setText("Ann. Return: ...")
        self.lbl_maxdd.setText("Max DD: ...")
        self.lbl_pairs_found.setText("Pairs Found: ...")
        
        QApplication.processEvents()

    def run_backtest(self):
        if self.strategy.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first (CSV or Yahoo Finance)")
            return
        
        if hasattr(self, 'worker') and self.worker is not None:
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
        
        self.btn_run.setEnabled(False)
        self.btn_optimize.setEnabled(False)
        self.reset_ui()
        
        params = {
            'beta_window': self.spin_beta.value(),
            'z_window': self.spin_z.value(),
            'entry_z': self.spin_entry.value(),
            'exit_z': self.spin_exit.value(),
            'stop_loss_z': self.spin_stop.value(),
            'tx_cost': self.spin_cost.value(),
            'slippage': self.spin_slip.value(),
            'risk_parity': True,
            'risk_parity_window': 60,
            'max_pair_weight': 0.3,
            'enable_cooldown': self.chk_enable_cooldown.isChecked(),
            'cooldown_days': self.spin_cooldown_days.value()
        }
        
        use_walk_forward = self.chk_walk_forward.isChecked()
        
        self.worker = BacktestWorker(self.strategy, params, use_walk_forward)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.result_signal.connect(self.display_results)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.finished.connect(lambda: self.btn_optimize.setEnabled(True))
        self.worker.start()

    def run_optimization(self):
        if self.strategy.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return
        
        metric = self.combo_metric.currentText()
        mode = self.combo_mode.currentText()
        
        if hasattr(self, 'opt_worker') and self.opt_worker is not None:
            if self.opt_worker.isRunning():
                self.opt_worker.terminate()
                self.opt_worker.wait()
        
        self.btn_optimize.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.reset_ui()
        
        if "Quick" in mode:
            param_grid = {
                'beta_window': [60, 252],
                'z_window': [30, 60],
                'entry_z': [2.0, 2.5],
                'exit_z': [0.0, 0.5],
                'stop_loss_z': [4.5]
            }
        else:
            param_grid = {
                'beta_window': [60, 126, 252, 504],
                'z_window': [20, 40, 60, 90],
                'entry_z': [1.0, 1.5, 2.0, 2.5],
                'exit_z': [-1.0, -0.5, 0.0, 0.5, 1.0],
                'stop_loss_z': [3.0, 4.5, 6.0]
            }

        fixed_costs = {
            'tx_cost': self.spin_cost.value(),
            'slippage': self.spin_slip.value()
        }

        self.opt_worker = OptimizerWorker(self.strategy, param_grid, metric, fixed_costs)
        self.opt_worker.progress_signal.connect(self.progress.setValue)
        self.opt_worker.log_signal.connect(lambda msg: self.statusBar().showMessage(msg))
        self.opt_worker.result_signal.connect(self.apply_best_parameters)
        self.opt_worker.finished_signal.connect(lambda: self.btn_optimize.setEnabled(True))
        self.opt_worker.finished_signal.connect(lambda: self.btn_run.setEnabled(True))
        self.opt_worker.start()

    def apply_best_parameters(self, best_params, best_score, all_results):
        """Callback when optimization finishes - shows detailed grid search results."""
        if not best_params:
            QMessageBox.warning(self, "Optimization Failed", "No profitable parameters found.")
            return
        
        # 1. Update UI inputs with best found values
        self.spin_beta.setValue(best_params['beta_window'])
        self.spin_z.setValue(best_params['z_window'])
        self.spin_entry.setValue(best_params['entry_z'])
        self.spin_exit.setValue(best_params['exit_z'])
        self.spin_stop.setValue(best_params['stop_loss_z'])
        
        # 2. Build detailed message with grid search statistics
        metric_name = self.combo_metric.currentText()
        
        # Calculate statistics from all results
        valid_scores = [s for s in all_results.values() if s > -999.0]
        if valid_scores:
            avg_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            min_score = np.min(valid_scores)
            max_score = np.max(valid_scores)
            
            stats_text = (
                f"Grid Search Statistics:\n"
                f"  • Combinations tested: {len(all_results)}\n"
                f"  • Valid results: {len(valid_scores)}\n"
                f"  • Mean {metric_name}: {avg_score:.4f}\n"
                f"  • Std Dev: {std_score:.4f}\n"
                f"  • Range: [{min_score:.4f}, {max_score:.4f}]\n\n"
            )
        else:
            stats_text = f"Grid Search Statistics:\n  • No valid results found\n\n"
        
        # Format best parameters
        params_text = (
            f"Best Parameters Found:\n"
            f"  • Beta Window: {best_params['beta_window']} days\n"
            f"  • Z-Score Window: {best_params['z_window']} days\n"
            f"  • Entry Threshold: {best_params['entry_z']:.2f} σ\n"
            f"  • Exit Threshold: {best_params['exit_z']:.2f} σ\n"
            f"  • Stop Loss: {best_params['stop_loss_z']:.2f} σ\n\n"
        )
        
        result_text = f"Best {metric_name}: {best_score:.4f}"
        
        # 3. Display comprehensive results
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Optimization Complete")
        msg.setText(f"Optimization finished!\n\n{result_text}\n\nParameters have been updated.")
        msg.setDetailedText(stats_text + params_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        
        # 4. Run visualization with new parameters
        self.run_backtest()

    def handle_error(self, error_msg):
        self.btn_run.setEnabled(True)
        self.btn_optimize.setEnabled(True)
        self.progress.setValue(0)
        QMessageBox.critical(self, "Simulation Error", f"An error occurred:\n{error_msg}")

    def display_results(self, cum_ret, pair_data, port_ret, mc_sims, extra_metrics, pairs):
        self.pair_data_cache = pair_data
        self.cointegrated_pairs = pairs
        
        # Update pairs found
        self.lbl_pairs_found.setText(f"Pairs Found: {len(pairs)}")
        
        # Populate Table
        self.table.setRowCount(len(pair_data))
        sorted_pairs = sorted(pair_data.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        
        for i, (name, data) in enumerate(sorted_pairs):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem(f"{data['sharpe']:.2f}"))
            total_ret = (1 + data['pnl']).cumprod().iloc[-1] - 1
            self.table.setItem(i, 2, QTableWidgetItem(f"{total_ret*100:.1f}%"))
            
            # Display stop losses for this pair
            sl_count = data.get('stop_losses', 0)
            sl_item = QTableWidgetItem(str(sl_count))
            if sl_count > 5:
                sl_item.setForeground(QColor('#ff4444'))  # Red if many stop losses
            elif sl_count > 2:
                sl_item.setForeground(QColor('#ffaa00'))  # Orange if moderate
            else:
                sl_item.setForeground(QColor('#00ff00'))  # Green if few
            self.table.setItem(i, 3, sl_item)
            
            status = "Profitable" if data['sharpe'] > 0 else "Loss"
            item = QTableWidgetItem(status)
            item.setForeground(QColor('#00ff00') if status == "Profitable" else QColor('#ff4444'))
            self.table.setItem(i, 4, item)

        # Equity Chart
        self.fig_equity.clear()
        ax = self.fig_equity.add_subplot(111)
        ax.set_facecolor('#2d2d2d')
        ax.grid(color='#444', linestyle='--', alpha=0.3)
        ax.tick_params(colors='white')
        
        # Plot Monte Carlo
        for sim in mc_sims:
            ax.plot(sim, color='gray', alpha=0.1, linewidth=0.5)
        
        # Plot Strategy
        ax.plot(cum_ret.values, color='#0d6efd', linewidth=2, label='Strategy', zorder=10)
        
        # Mark train/test split if applicable
        split_idx = None
        if extra_metrics.get('split_date'):
            split_idx = cum_ret.index.get_loc(extra_metrics['split_date'])
            ax.axvline(split_idx, color='orange', linestyle='--', linewidth=2, label='Train/Test Split')
        
        ax.set_title("Portfolio Equity vs Monte Carlo", color='white', fontsize=12)
        ax.set_xlabel("Days", color='white')
        ax.set_ylabel("Cumulative Return", color='white')
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
        self.canvas_equity.draw()

        # FIX: Calculate Metrics ONLY on test period if walk-forward is enabled
        if extra_metrics.get('split_date') and self.chk_walk_forward.isChecked():
            # Use only test period for metrics
            test_port_ret = port_ret.loc[extra_metrics['split_date']:]
            test_cum_ret = cum_ret.loc[extra_metrics['split_date']:]
            metric_label = "Test Period"
        else:
            # Use full period
            test_port_ret = port_ret
            test_cum_ret = cum_ret
            metric_label = "Full Period"
        
        # Calculate Metrics on selected period
        ann_ret = test_port_ret.mean() * 252
        ann_vol = test_port_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        
        downside_ret = test_port_ret[test_port_ret < 0]
        downside_std = downside_ret.std() * np.sqrt(252)
        sortino = ann_ret / downside_std if downside_std != 0 else 0
        
        dd = (test_cum_ret / test_cum_ret.cummax()) - 1
        max_dd = dd.min()

        # Display with label indicating which period
        self.lbl_sharpe.setText(f"Sharpe Ratio ({metric_label}): {sharpe:.2f}")
        self.lbl_sortino.setText(f"Sortino Ratio ({metric_label}): {sortino:.2f}")
        self.lbl_cagr.setText(f"Ann. Return ({metric_label}): {ann_ret*100:.2f}%")
        self.lbl_maxdd.setText(f"Max Drawdown ({metric_label}): {max_dd*100:.2f}%")

        nb_trades = extra_metrics['total_trades']
        avg_ret = extra_metrics['avg_trade_ret']
        avg_cost = extra_metrics.get('avg_cost_per_trade', 0)
        stop_losses = extra_metrics.get('total_stop_losses', 0)
        
        self.lbl_trades.setText(f"Total Trades: {nb_trades}")
        self.lbl_avg_ret.setText(f"Avg Ret/Trade: {avg_ret*100:.3f}%")
        self.lbl_avg_cost.setText(f"Avg Cost/Trade: {avg_cost*100:.3f}%")
        self.lbl_stop_losses.setText(f"Stop Losses Hit: {stop_losses}")

    def drill_down_pair(self, item):
        row = item.row()
        pair_name = self.table.item(row, 0).text()
        data = self.pair_data_cache.get(pair_name)
        if not data:
            return

        self.tab_charts.setCurrentIndex(1)
        z = data['z']
        pos = data['pos']
        
        self.fig_detail.clear()
        ax = self.fig_detail.add_subplot(111)
        ax.set_facecolor('#2d2d2d')
        ax.grid(color='#444', linestyle='--', alpha=0.3)
        ax.tick_params(colors='white')

        # Plot Z-Score
        ax.plot(z.index, z.values, color='cyan', label='Z-Score', linewidth=1.5, alpha=0.8)
        
        # Threshold Lines
        ax.axhline(self.spin_entry.value(), color='red', linestyle='--', label='Entry (Short)', alpha=0.7)
        ax.axhline(-self.spin_entry.value(), color='green', linestyle='--', label='Entry (Long)', alpha=0.7)
        ax.axhline(self.spin_exit.value(), color='yellow', linestyle=':', label='Exit (TP)', alpha=0.7)
        ax.axhline(-self.spin_exit.value(), color='yellow', linestyle=':', alpha=0.7)
        ax.axhline(self.spin_stop.value(), color='orange', linestyle='-.', label='Stop Loss', alpha=0.7)
        ax.axhline(-self.spin_stop.value(), color='orange', linestyle='-.', alpha=0.7)
        
        # Trade Markers
        buys = z.index[(pos == 1) & (pos.shift(1) != 1)]
        sells = z.index[(pos == -1) & (pos.shift(1) != -1)]
        
        if len(buys) > 0:
            ax.scatter(buys, z.loc[buys], color='lime', marker='^', s=100, label='Long Entry', zorder=5, edgecolors='black')
        if len(sells) > 0:
            ax.scatter(sells, z.loc[sells], color='red', marker='v', s=100, label='Short Entry', zorder=5, edgecolors='black')

        ax.set_title(f"Z-Score Analysis: {pair_name}", color='white', fontsize=12)
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Z-Score", color='white')
        ax.legend(loc='best', facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
        self.canvas_detail.draw()

    def load_yf(self):
        dialog = YFinanceDialog(self)
        if dialog.exec():
            tickers, start = dialog.get_data()
            try:
                self.strategy.fetch_yfinance(tickers, start)
                QMessageBox.information(self, "Success", f"Loaded {len(self.strategy.assets)} assets from Yahoo Finance.")
            except Exception as e:
                QMessageBox.critical(self, "Data Error", str(e))

    def load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if fname:
            try:
                self.strategy.load_csv(fname)
                QMessageBox.information(self, "Success", f"Loaded {len(self.strategy.assets)} assets from CSV.")
            except Exception as e:
                QMessageBox.critical(self, "File Error", str(e))

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON (*.json)")
        if path:
            cfg = {
                "beta": self.spin_beta.value(),
                "z": self.spin_z.value(),
                "entry": self.spin_entry.value(),
                "exit": self.spin_exit.value(),
                "stop": self.spin_stop.value(),
                "cost": self.spin_cost.value(),
                "slip": self.spin_slip.value(),
                "walk_forward": self.chk_walk_forward.isChecked(),
                "enable_cooldown": self.chk_enable_cooldown.isChecked(),  # NEW
                "cooldown_days": self.spin_cooldown_days.value()  # NEW
            }
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if path:
            with open(path, 'r') as f:
                cfg = json.load(f)
            self.spin_beta.setValue(cfg.get('beta', 252))
            self.spin_z.setValue(cfg.get('z', 60))
            self.spin_entry.setValue(cfg.get('entry', 2.0))
            self.spin_exit.setValue(cfg.get('exit', 0.0))
            self.spin_stop.setValue(cfg.get('stop', 4.5))
            self.spin_cost.setValue(cfg.get('cost', 0.001))
            self.spin_slip.setValue(cfg.get('slip', 0.0005))
            self.chk_walk_forward.setChecked(cfg.get('walk_forward', True))
            self.chk_enable_cooldown.setChecked(cfg.get('enable_cooldown', False))  # NEW
            self.spin_cooldown_days.setValue(cfg.get('cooldown_days', 20))  # NEW

    def load_settings(self):
        geom = self.settings.value("geometry")
        state = self.settings.value("windowState")
        if geom:
            self.restoreGeometry(geom)
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
