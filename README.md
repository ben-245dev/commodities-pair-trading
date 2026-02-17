# commodities-pair-trading
Desktop application for Statistical Arbitrage and Pairs Trading. Built with Python &amp; PyQt6, featuring Rolling OLS, Monte Carlo stress testing, and an interactive dark-mode dashboard.


A desktop application designed for backtesting and analyzing Market Neutral (Pairs Trading) strategies. Built with a focus on analysis, it combines rigorous statistical methods with a modern, dockable GUI to visualize performance, risk, and trade micro-structure.
Key Features

Advanced Math Engine:

        Automated Cointegration scanning (Engle-Granger / ADF Tests).

        Dynamic Hedge Ratios using Rolling OLS.

        Configurable Z-Score entry/exit/stop-loss logic.

Professional GUI (PyQt6):

        Dark theme with dockable/floating widgets.

        Interactive drill-down: Click a pair to see specific trade markers on the spread.

        Real-time "Tear Sheet" metrics (Sharpe, Sortino, CAGR, Max Drawdown).

Risk Management:

        Monte Carlo Simulation: Bootstrap resampling to distinguish skill from luck.

        Risk Parity: Inverse volatility weighting for portfolio construction.

        Cost Modeling: Realistic simulation of Slippage and Commissions.

Production:

        Walk forward analysis to avoid look ahead bias.

        Cooldown to avoid trading bankrupting securities.

Data Flexibility:

        Built-in Yahoo Finance downloader.

        Support for custom CSV imports.

        Save/Load strategy configurations (.json).

Tech Stack
    
    Core: Python 3.x
    
    UI: PyQt6 (Qt for Python)
    
    Quant: Pandas, NumPy, Statsmodels, SciPy
    
    Visualization: Matplotlib (embedded), YFinance
