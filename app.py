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
    QTabWidget, QInputDialog, QDialog, QDialogButtonBox, QLineEdit
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
        self.txt_tickers = QLineEdit("GLD, GDX, SLV, SIL, COPX, FCX, SCCO, BHP, RIO")
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

# --- CORE TRADING ENGINE ---
class TradingStrategy:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.assets: List[str] = []

    def fetch_yfinance(self, tickers: List[str], start_date: str):
        """Downloads historical data using YFinance."""
        print(f"Downloading {tickers} from {start_date}...")
        try:
            data = yf.download(tickers, start=start_date, progress=False)['Close']
            if data.empty: raise ValueError("No data returned from Yahoo Finance.")
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

    def run_monte_carlo(self, returns_series: pd.Series, simulations: int = 50):
        """
        Runs Monte Carlo simulations to stress-test the strategy.
        Aligned to the exact duration of the backtest.
        """
        sim_results = []
        daily_rets = returns_series.pct_change().dropna().values
        horizon = len(daily_rets)
        
        # Calculate offset to align the simulation graph with the strategy warmup period
        start_offset = len(returns_series) - horizon

        for _ in range(simulations):
            # Bootstrap resampling with replacement
            shuffled_rets = np.random.choice(daily_rets, size=horizon, replace=True)
            
            # Reconstruct equity curve (base 1.0)
            cum_ret_sim = np.cumprod(1 + shuffled_rets)
            
            # Prepend ones to match the x-axis of the main strategy
            full_sim_curve = np.concatenate(([1.0] * start_offset, cum_ret_sim))
            sim_results.append(full_sim_curve)
        
        return sim_results

    def backtest_pair(self, s1, s2, params):
        """
        Backtests a single pair strategy.
        Returns: PnL Series, Z-Score Series, Positions Series
        """
        # 1. Prepare Data
        log_df = np.log(self.df)
        y, x = log_df[s1], sm.add_constant(log_df[s2])

        # 2. Rolling OLS (Hedge Ratio)
        model = RollingOLS(y, x, window=params['beta_window'])
        rolling_res = model.fit()
        res_params = rolling_res.params.shift(1) # Avoid look-ahead bias
        beta, alpha = res_params[s2], res_params['const']

        # 3. Z-Score Calculation
        spread = y - (beta * log_df[s2] + alpha)
        roll_mean = spread.rolling(window=params['z_window']).mean()
        roll_std = spread.rolling(window=params['z_window']).std()
        z_score = (spread - roll_mean) / roll_std

        # 4. Generate Signals
        signals = pd.Series(0, index=z_score.index)
        signals[z_score > params['entry_z']] = -1  # Short Spread
        signals[z_score < -params['entry_z']] = 1  # Long Spread
        
        # 5. Position Management
        positions = signals.replace(0, np.nan).ffill()
        exit_mask = (
            ((positions == -1) & (z_score < params['exit_z'])) |
            ((positions == 1) & (z_score > -params['exit_z']))
        )
        positions[exit_mask] = 0
        # Stop Loss
        positions[abs(z_score) > params['stop_loss_z']] = 0
        positions = positions.fillna(0)

        # 6. Calculate PnL
        pos_delayed = positions.shift(1) # Enter trade on next open
        ret1 = self.df[s1].pct_change()
        ret2 = self.df[s2].pct_change()
        spread_ret = ret1 - beta * ret2
        
        pnl = pos_delayed * spread_ret
        
        # 7. Apply Costs (Commission + Slippage)
        trades = positions.diff().abs()
        total_cost = params['tx_cost'] + params['slippage']
        costs = trades * total_cost
        
        net_pnl = pnl - costs
        return net_pnl.fillna(0), z_score, positions

# --- WORKER THREAD (BACKGROUND TASKS) ---
class BacktestWorker(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(object, object, object, object)
    error_signal = pyqtSignal(str)

    def __init__(self, strategy, params):
        super().__init__()
        self.strategy = strategy
        self.params = params

    def run(self):
        try:
            # 1. Cointegration Scan (Engle-Granger)
            log_df = np.log(self.strategy.df)
            corr = self.strategy.df.corr()
            pairs = []
            assets = self.strategy.assets
            n = len(assets)
            
            for i in range(n):
                for j in range(i + 1, n):
                    s1, s2 = assets[i], assets[j]
                    if corr.loc[s1, s2] < 0.85: continue # Correlation filter
                    
                    res = linregress(log_df[s2], log_df[s1])
                    spread = log_df[s1] - (res.slope * log_df[s2] + res.intercept)
                    if adfuller(spread)[1] < 0.05: # ADF Test
                        pairs.append((s1, s2))

            # 2. Backtest Loop
            all_pnls = pd.DataFrame(index=self.strategy.df.index)
            pair_data = {} 
            
            for i, (s1, s2) in enumerate(pairs):
                pnl, z, pos = self.strategy.backtest_pair(s1, s2, self.params)
                
                # Store results if data exists (even if losing)
                if pnl.std() != 0:
                    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252)
                    name = f"{s1}-{s2}"
                    all_pnls[name] = pnl
                    pair_data[name] = {'sharpe': sharpe, 'z': z, 'pos': pos, 'pnl': pnl}
                
                self.progress_signal.emit(int((i+1)/len(pairs)*100))

            if all_pnls.empty:
                self.error_signal.emit("No valid pairs found. Try relaxing filters.")
                return

            # 3. Portfolio Construction (Risk Parity)
            final_pnls = all_pnls.fillna(0)
            if self.params['risk_parity']:
                # Inverse Volatility Weighting
                inv_vol = 1.0 / final_pnls.rolling(60).std().shift(1).replace(0, np.nan)
                weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
                port_ret = (final_pnls * weights).sum(axis=1)
            else:
                # Equal Weighting
                port_ret = final_pnls.mean(axis=1)

            cum_ret = (1 + port_ret).cumprod()
            
            # 4. Run Monte Carlo
            mc_sims = self.strategy.run_monte_carlo(cum_ret, simulations=50)

            self.result_signal.emit(cum_ret, pair_data, port_ret, mc_sims)

        except Exception as e:
            self.error_signal.emit(str(e))

# --- MAIN APPLICATION WINDOW ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statistical Arbitrage app")
        self.resize(1600, 900)
        self.setStyleSheet(DARK_THEME_QSS)
        
        self.strategy = TradingStrategy()
        self.pair_data_cache = {} 
        self.init_ui()
        
        self.settings = QSettings("Arbitrage startegies", "ben-245dev")
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
        
        self.layout_controls.addRow(QLabel("--- Rolling Window Parameters ---"))
        self.spin_beta = QSpinBox(); self.spin_beta.setRange(1, 100_000); self.spin_beta.setValue(252)
        self.layout_controls.addRow("Beta Window (Days):", self.spin_beta)
        
        self.spin_z = QSpinBox(); self.spin_z.setRange(1, 100_000); self.spin_z.setValue(60)
        self.layout_controls.addRow("Z-Score Window (Days):", self.spin_z)
        
        self.layout_controls.addRow(QLabel("--- Trading Logic (Sigma) ---"))
        self.spin_entry = QDoubleSpinBox(); self.spin_entry.setValue(2.0);self.spin_entry.setRange(-100_000, 100_000); self.spin_entry.setSingleStep(0.1)
        self.layout_controls.addRow("Entry Threshold:", self.spin_entry)
        
        self.spin_exit = QDoubleSpinBox(); self.spin_exit.setValue(0.0); self.spin_exit.setRange(-100_000, 100_000); self.spin_exit.setSingleStep(0.1)
        self.layout_controls.addRow("Exit Threshold (TP):", self.spin_exit)
        
        self.spin_stop = QDoubleSpinBox(); self.spin_stop.setValue(4.5); self.spin_stop.setRange(-100_000, 100_000); self.spin_stop.setSingleStep(0.1)
        self.layout_controls.addRow("Stop Loss:", self.spin_stop)
        
        self.layout_controls.addRow(QLabel("--- Execution Costs ---"))
        self.spin_cost = QDoubleSpinBox(); self.spin_cost.setValue(0.001);self.spin_cost.setRange(-100_000, 100_000); self.spin_cost.setDecimals(4)
        self.layout_controls.addRow("Commission (%):", self.spin_cost)
        self.spin_slip = QDoubleSpinBox(); self.spin_slip.setValue(0.0005);self.spin_slip.setRange(-100_000, 100_000); self.spin_slip.setDecimals(4)
        self.layout_controls.addRow("Slippage (%):", self.spin_slip)
        
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setStyleSheet("background-color: #198754; font-size: 14px; height: 40px;")
        self.btn_run.clicked.connect(self.run_backtest)
        self.layout_controls.addRow(self.btn_run)
        
        # Config Section
        save_load_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save Config"); self.btn_save.clicked.connect(self.save_config)
        self.btn_load_cfg = QPushButton("Load Config"); self.btn_load_cfg.clicked.connect(self.load_config)
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
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Pair", "Sharpe", "Total Return", "Status"])
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
        self.metrics_layout.addWidget(self.lbl_sharpe)
        self.metrics_layout.addWidget(self.lbl_sortino)
        self.metrics_layout.addWidget(self.lbl_cagr)
        self.metrics_layout.addWidget(self.lbl_maxdd)
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

    # --- RESET UI LOGIC (FIX FOR VISUAL PERSISTENCE) ---
    def reset_ui(self):
        """Clears all charts and tables to indicate a new run."""
        self.table.setRowCount(0)
        
        # Clear Equity Chart
        self.fig_equity.clear()
        ax = self.fig_equity.add_subplot(111)
        ax.set_facecolor('#2d2d2d')
        ax.text(0.5, 0.5, "Running Simulation...", color='white', ha='center')
        ax.axis('off')
        self.canvas_equity.draw()
        
        # Clear Details
        self.fig_detail.clear()
        self.canvas_detail.draw()
        
        # Reset Metrics
        self.lbl_sharpe.setText("Sharpe: ...")
        self.lbl_sortino.setText("Sortino: ...")
        self.lbl_cagr.setText("Ann. Return: ...")
        self.lbl_maxdd.setText("Max DD: ...")
        
        QApplication.processEvents()

    # --- MAIN EXECUTION ---
    def run_backtest(self):
        # 1. Kill existing thread if running (Fix for stuck UI)
        if hasattr(self, 'worker') and self.worker is not None:
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
        
        # 2. Disable UI & Reset
        self.btn_run.setEnabled(False)
        self.reset_ui()
        
        # 3. Gather Parameters
        params = {
            'beta_window': self.spin_beta.value(),
            'z_window': self.spin_z.value(),
            'entry_z': self.spin_entry.value(),
            'exit_z': self.spin_exit.value(),
            'stop_loss_z': self.spin_stop.value(),
            'tx_cost': self.spin_cost.value(),
            'slippage': self.spin_slip.value(),
            'risk_parity': True
        }
        
        # 4. Start Worker
        self.worker = BacktestWorker(self.strategy, params)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.result_signal.connect(self.display_results)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.start()

    def handle_error(self, error_msg):
        self.btn_run.setEnabled(True)
        self.progress.setValue(0)
        QMessageBox.critical(self, "Simulation Error", f"An error occurred:\n{error_msg}")

    def display_results(self, cum_ret, pair_data, port_ret, mc_sims):
        self.pair_data_cache = pair_data 
        
        # 1. Populate Table
        self.table.setRowCount(len(pair_data))
        sorted_pairs = sorted(pair_data.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        
        for i, (name, data) in enumerate(sorted_pairs):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem(f"{data['sharpe']:.2f}"))
            total_ret = (1 + data['pnl']).cumprod().iloc[-1] - 1
            self.table.setItem(i, 2, QTableWidgetItem(f"{total_ret*100:.1f}%"))
            
            status = "Profitable" if data['sharpe'] > 0 else "Loss"
            item = QTableWidgetItem(status)
            item.setForeground(QColor('#00ff00') if status == "Profitable" else QColor('#ff4444'))
            self.table.setItem(i, 3, item)

        # 2. Equity Chart with Monte Carlo
        self.fig_equity.clear()
        ax = self.fig_equity.add_subplot(111)
        ax.set_facecolor('#2d2d2d')
        ax.grid(color='#444', linestyle='--')
        ax.tick_params(colors='white')
        
        # Plot Monte Carlo (Gray background)
        for sim in mc_sims:
            ax.plot(np.arange(len(sim)), sim, color='gray', alpha=0.1)
            
        # Plot Strategy (Blue)
        ax.plot(np.arange(len(cum_ret)), cum_ret.values, color='#0d6efd', linewidth=2, label="Strategy")
        ax.set_title("Portfolio Equity vs Monte Carlo Stress Test", color='white')
        ax.legend(loc='upper left')
        self.canvas_equity.draw()

        # 3. Calculate Metrics
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        
        downside_ret = port_ret[port_ret < 0]
        downside_std = downside_ret.std() * np.sqrt(252)
        sortino = ann_ret / downside_std if downside_std != 0 else 0
        
        dd = (cum_ret / cum_ret.cummax()) - 1
        max_dd = dd.min()

        self.lbl_sharpe.setText(f"Sharpe Ratio: {sharpe:.2f}")
        self.lbl_sortino.setText(f"Sortino Ratio: {sortino:.2f}")
        self.lbl_cagr.setText(f"Ann. Return: {ann_ret*100:.2f}%")
        self.lbl_maxdd.setText(f"Max Drawdown: {max_dd*100:.2f}%")

    def drill_down_pair(self, item):
        row = item.row()
        pair_name = self.table.item(row, 0).text()
        data = self.pair_data_cache.get(pair_name)
        if not data: return

        self.tab_charts.setCurrentIndex(1)
        z = data['z']
        pos = data['pos']
        
        self.fig_detail.clear()
        ax = self.fig_detail.add_subplot(111)
        ax.set_facecolor('#2d2d2d')
        ax.grid(color='#444', linestyle='--')
        ax.tick_params(colors='white')

        # Plot Z-Score
        ax.plot(z.index, z.values, color='cyan', label='Z-Score', linewidth=1)
        
        # Threshold Lines
        ax.axhline(self.spin_entry.value(), color='red', linestyle='--')
        ax.axhline(-self.spin_entry.value(), color='green', linestyle='--')
        ax.axhline(self.spin_exit.value(), color='yellow', linestyle=':', label='Exit (TP)')
        ax.axhline(-self.spin_exit.value(), color='yellow', linestyle=':')
        
        # Trade Markers
        buys = z.index[(pos == 1) & (pos.shift(1) != 1)]
        sells = z.index[(pos == -1) & (pos.shift(1) != -1)]
        
        ax.scatter(buys, z.loc[buys], color='lime', marker='^', s=100, label='Long Entry', zorder=5)
        ax.scatter(sells, z.loc[sells], color='red', marker='v', s=100, label='Short Entry', zorder=5)

        ax.set_title(f"Micro-Structure: {pair_name}", color='white')
        ax.legend()
        self.canvas_detail.draw()

    # --- DATA & CONFIG HANDLERS ---
    def load_yf(self):
        dialog = YFinanceDialog(self)
        if dialog.exec():
            tickers, start = dialog.get_data()
            try:
                self.strategy.fetch_yfinance(tickers, start)
                QMessageBox.information(self, "Success", f"Loaded {len(self.strategy.assets)} assets.")
            except Exception as e:
                QMessageBox.critical(self, "Data Error", str(e))

    def load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if fname:
            try:
                self.strategy.load_csv(fname)
                QMessageBox.information(self, "Success", f"Loaded {len(self.strategy.assets)} assets.")
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
                "slip": self.spin_slip.value()
            }
            with open(path, 'w') as f: json.dump(cfg, f)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if path:
            with open(path, 'r') as f: cfg = json.load(f)
            self.spin_beta.setValue(cfg.get('beta', 252))
            self.spin_z.setValue(cfg.get('z', 60))
            self.spin_entry.setValue(cfg.get('entry', 2.0))
            self.spin_exit.setValue(cfg.get('exit', 0.0))
            self.spin_stop.setValue(cfg.get('stop', 4.5))
            self.spin_cost.setValue(cfg.get('cost', 0.001))
            self.spin_slip.setValue(cfg.get('slip', 0.0005))

    def load_settings(self):
        geom = self.settings.value("geometry")
        state = self.settings.value("windowState")
        if geom: self.restoreGeometry(geom)
        if state: self.restoreState(state)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
