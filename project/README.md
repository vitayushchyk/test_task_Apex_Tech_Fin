# Technical test task for Apex Tech Fin

This project is designed for backtesting algorithmic trading strategies using Python. The system supports multiple strategies, such as **SMA Crossover**, **RSI + Bollinger Bands**, and **VWAP Reversion**, with the ability to evaluate their performance through backtests.

---

## Features
- **Support for multiple trading strategies**:
  - **SMA Crossover Strategy**: Uses short and long simple moving averages for generating buy and sell signals.
  - **RSI + Bollinger Bands**: Combines the Relative Strength Index (RSI) with Bollinger Band indicators for signal generation.
  - **VWAP Reversion Strategy**: Based on price deviation from the Volume-Weighted Average Price (VWAP) to identify potential reversals.
- **Real market data** is used to generate signals.
- Results are saved as CSV files and visualized in PNG graphs.
- Metrics including **Total Return**, **Sharpe Ratio**, **Max Drawdown**, and more are automatically calculated.

---

## Results

### Key Metrics
*The strategies are further visualized with graphs (saved as PNG images).*

---

## Installation

1. **Clone the repository**:
    ```bash
    git clone git@github.com:vitayushchyk/test_task_Apex_Tech_Fin.git
    cd project
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. **Run the backtester**:
    Use `main.py` to execute all strategies:
    ```bash
    python main.py
    ```