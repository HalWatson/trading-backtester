# Python Trading Strategy Backtester

## Overview

An independent quantitative finance project developing a Python-based framework for testing trading strategies on historical market data.

The initial version will evaluate a moving-average crossover strategy on daily Nasdaq-100 ETF data, measuring performance, transaction costs and risk.

## Objectives

- Retrieve and validate historical market data.
- Generate trading signals from predefined rules.
- Simulate portfolio returns with realistic execution timing.
- Incorporate transaction costs into performance calculations.
- Measure returns, volatility, win rate and maximum drawdown.
- Compare performance against a benchmark.
- Evaluate robustness using out-of-sample testing and parameter sensitivity.

## Initial Strategy

A long-only moving-average crossover strategy using 20-day and 50-day simple moving averages.

The strategy enters a long position when the fast moving average is above the slow moving average and otherwise remains in cash. Positions will be lagged to avoid look-ahead bias.

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- yfinance


## Data

- **Instrument:** Invesco QQQ ETF (ticker: QQQ)
- **Data source:** Yahoo Finance, accessed through the `yfinance` Python library
- **Frequency:** Daily
- **Start date:** 1 January 2010, subject to data availability
- **Price adjustment:** `auto_adjust=True`, which adjusts historical OHLC prices for relevant corporate actions
- **Local dataset:** `data/qqq_daily.csv`
- **Download date:** Tuesday 13th October 2026
- **Observations:** 10

The dataset will be checked for missing values, duplicate dates, unexpected prices and other data-quality issues before being used in the backtest.

QQQ is an ETF proxy for Nasdaq-100 exposure. Results may differ from those of Nasdaq-100 index products and futures because of differences in trading mechanics, costs and execution.

## Planned Performance Metrics

- Total and annualised return
- Annualised volatility
- Sharpe ratio
- Maximum drawdown
- Win rate
- Number of trades
- Transaction costs
- Benchmark-relative performance

## Project Structure

- `data/`: historical data
- `notebooks/`: exploration and experiments
- `src/`: reusable backtesting modules
- `tests/`: validation and unit tests
- `results/`: performance outputs and charts

## Limitations

Backtest results are hypothetical and depend on data quality, assumptions, transaction costs and execution timing. Historical performance does not guarantee future results. QQQ ETF results may not transfer directly to Nasdaq-100 futures.

## Status

Day 1: project setup and research specification.


