
import pandas as pd

# Load the historical QQQ dataset
data = pd.read_csv(
    "data/qqq_daily.csv",
    index_col=0,
    parse_dates=True
)

# Ensure the index is a proper datetime index
data.index = pd.to_datetime(data.index, utc=True).tz_convert(
    "America/New_York"
)



# --------------------------------------------------
# DAY 3: DATA QUALITY VALIDATION
# --------------------------------------------------

print("\n--- Data Quality Checks ---")

# 1. Check for duplicate dates
duplicate_dates = data.index.duplicated().sum()
print(f"Duplicate dates: {duplicate_dates}")

# 2. Check whether dates are in chronological order
dates_ordered = data.index.is_monotonic_increasing
print(f"Dates in chronological order: {dates_ordered}")

# 3. Check for missing closing prices
missing_closes = data["Close"].isna().sum()
print(f"Missing closing prices: {missing_closes}")

# 4. Check for invalid closing prices
invalid_closes = (
    (data["Close"] <= 0) | (~data["Close"].map(pd.notna))
).sum()
print(f"Invalid closing prices: {invalid_closes}")

# 5. Check whether the dataset contains any rows
print(f"Number of rows: {len(data)}")

# Stop the program if the data fails essential checks
assert len(data) > 0, "Dataset is empty!"
assert duplicate_dates == 0, "Duplicate dates found!"
assert dates_ordered, "Dates are not chronological!"
assert missing_closes == 0, "Missing closing prices found!"
assert invalid_closes == 0, "Invalid closing prices found!"

print("All essential data quality checks passed!")



# Select the daily closing prices
prices = data["Close"]

# Inspect the available date range
print("First date:", prices.index.min().date())
print("Last date:", prices.index.max().date())




# Calculate the 20-day and 50-day simple moving averages
sma_20 = prices.rolling(window=20).mean()
sma_50 = prices.rolling(window=50).mean()



print("First 10 moving-average values:")
print(pd.DataFrame({
    "Close": prices,
    "SMA_20": sma_20,
    "SMA_50": sma_50
}).head(60))

print("\nMissing values in each series:")
print(pd.DataFrame({
    "Close": prices,
    "SMA_20": sma_20,
    "SMA_50": sma_50
}).isnull().sum())






# Generate the trading signal

# Generate signals only when both moving averages exist
signal = pd.Series(float("nan"), index=prices.index)

valid = sma_50.notna()

signal.loc[valid] = (
    sma_20.loc[valid] > sma_50.loc[valid]
).astype(int)

daily_returns = prices.pct_change()

# Apply yesterday's signal to today's market return
position = signal.shift(1)

# Calculate the strategy's daily returns
strategy_returns = position * daily_returns



# Calculate changes in position, treating the initial position as cash
position_change = position.diff().abs()

# Charge the initial entry trade when the first valid position is established
first_valid_position = position.first_valid_index()

if first_valid_position is not None:
    position_change.loc[first_valid_position] = abs(
        position.loc[first_valid_position]
    )

# Assume transaction costs of 0.1% per position change
transaction_cost = position_change * 0.001


# Deduct transaction costs from strategy returns
strategy_returns_after_costs = (
    strategy_returns - transaction_cost
)


# --------------------------------------------------
# DAY 3: STRATEGY CALCULATION VALIDATION
# --------------------------------------------------

print("\n--- Strategy Calculation Checks ---")

# 1. Positions must be either cash (0) or invested (1)
valid_positions = position.dropna().isin([0.0, 1.0]).all()
assert valid_positions, "Unexpected position value!"

# 2. Confirm that yesterday's signal determines today's position
expected_position = signal.shift(1)
assert position.equals(expected_position), "Position lag check failed!"

# 3. Confirm that transaction costs match position changes
expected_cost = position_change * 0.001
assert transaction_cost.equals(expected_cost), "Transaction cost check failed!"

# 4. Confirm that costs are deducted correctly
expected_returns = strategy_returns - transaction_cost

assert strategy_returns_after_costs.equals(
    expected_returns
), "Net return calculation failed!"

print("PASS: Positions are valid.")
print("PASS: Signals are lagged by one trading day.")
print("PASS: Transaction costs match position changes.")
print("PASS: Returns after costs are calculated correctly.")
print("All strategy calculation checks passed!")



# Compare both approaches over the same evaluation period
evaluation_start = position.first_valid_index()

strategy_returns_test = strategy_returns_after_costs.loc[evaluation_start:]
market_returns_test = daily_returns.loc[evaluation_start:]

# Calculate cumulative growth over the shared period
strategy_growth = (1 + strategy_returns_test.fillna(0)).cumprod()
market_growth = (1 + market_returns_test.fillna(0)).cumprod()

strategy_cumulative_return = strategy_growth - 1
market_cumulative_return = market_growth - 1

# Calculate annualised returns
number_of_days = (
    strategy_returns_test.index[-1] - strategy_returns_test.index[0]
).days

number_of_years = number_of_days / 365.25

strategy_annualised_return = (
    strategy_growth.iloc[-1] ** (1 / number_of_years)
) - 1

market_annualised_return = (
    market_growth.iloc[-1] ** (1 / number_of_years)
) - 1

print("\nAnnualised returns:")
print(f"Strategy: {strategy_annualised_return:.2%}")
print(f"Buy and hold: {market_annualised_return:.2%}")

# Calculate annualised volatility
strategy_volatility = strategy_returns_test.std() * (252 ** 0.5)
market_volatility = market_returns_test.std() * (252 ** 0.5)

print("\nAnnualised volatility:")
print(f"Strategy: {strategy_volatility:.2%}")
print(f"Buy and hold: {market_volatility:.2%}")



# Calculate annualised Sharpe ratios using daily returns
strategy_sharpe = (
    strategy_returns_test.mean()
    / strategy_returns_test.std()
) * (252 ** 0.5)

market_sharpe = (
    market_returns_test.mean()
    / market_returns_test.std()
) * (252 ** 0.5)

print("\nAnnualised Sharpe ratios (zero risk-free rate):")
print(f"Strategy: {strategy_sharpe:.2f}")
print(f"Buy and hold: {market_sharpe:.2f}")





# Calculate maximum drawdown
strategy_running_peak = strategy_growth.cummax()
market_running_peak = market_growth.cummax()

strategy_drawdown = (
    strategy_growth / strategy_running_peak
) - 1

market_drawdown = (
    market_growth / market_running_peak
) - 1

strategy_max_drawdown = strategy_drawdown.min()
market_max_drawdown = market_drawdown.min()

print("\nMaximum drawdown:")
print(f"Strategy: {strategy_max_drawdown:.2%}")
print(f"Buy and hold: {market_max_drawdown:.2%}")


import matplotlib.pyplot as plt

# Plot cumulative returns for both approaches
plt.figure(figsize=(12, 6))

plt.plot(
    strategy_cumulative_return.index,
    strategy_cumulative_return * 100,
    label="20/50 SMA Strategy"
)

plt.plot(
    market_cumulative_return.index,
    market_cumulative_return * 100,
    label="Buy and Hold QQQ"
)

plt.title("QQQ Strategy vs Buy and Hold")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the chart to the results folder
plt.savefig("results/cumulative_returns.png", dpi=300)

plt.show()


print("\nFinal performance:")
print(f"Strategy cumulative return: "
      f"{strategy_cumulative_return.iloc[-1]:.2%}")
print(f"Buy-and-hold cumulative return: "
      f"{market_cumulative_return.iloc[-1]:.2%}")


print("\nLatest returns after transaction costs:")
print(pd.DataFrame({
    "Position": position,
    "Position_Change": position_change,
    "Transaction_Cost": transaction_cost,
    "Strategy_Return_Before_Costs": strategy_returns,
    "Strategy_Return_After_Costs": strategy_returns_after_costs
}).tail(15))


print("\nLatest strategy returns:")
print(pd.DataFrame({
    "Market_Return": daily_returns,
    "Signal": signal,
    "Position": position,
    "Strategy_Return": strategy_returns
}).tail(10))






print("\nLatest trading signals:")
print(pd.DataFrame({
    "Close": prices,
    "SMA_20": sma_20,
    "SMA_50": sma_50,
    "Signal": signal
}).tail(10))

print("\nSignal counts:")
print(signal.value_counts())





# Calculate daily returns from QQQ closing prices

print("\nLatest daily market returns:")
print(daily_returns.tail(10))





print("Index type:", type(prices.index))
print("Index example:", prices.index[:3])






# Out-of-sample test: January 2022 onwards
test_start = pd.Timestamp(
    "2022-01-01",
    tz="America/New_York"
)

# Select dates using a timezone-aware comparison
strategy_oos = strategy_returns_test.loc[
    strategy_returns_test.index >= test_start
]

market_oos = market_returns_test.loc[
    market_returns_test.index >= test_start
]

# Calculate cumulative growth during the test period
strategy_oos_growth = (
    1 + strategy_oos.fillna(0)
).cumprod()

market_oos_growth = (
    1 + market_oos.fillna(0)
).cumprod()

# Calculate total returns during the test period
strategy_oos_return = strategy_oos_growth.iloc[-1] - 1
market_oos_return = market_oos_growth.iloc[-1] - 1

print("\nOut-of-sample results: January 2022 onwards")
print(f"Strategy total return: {strategy_oos_return:.2%}")
print(f"Buy-and-hold total return: {market_oos_return:.2%}")




# Annualised out-of-sample returns
days = (
    strategy_oos.index[-1] - strategy_oos.index[0]
).days

years = days / 365.25

strategy_annual_return = (
    strategy_oos_growth.iloc[-1] ** (1 / years) - 1
)

market_annual_return = (
    market_oos_growth.iloc[-1] ** (1 / years) - 1
)

# Annualised volatility
strategy_annual_volatility = (
    strategy_oos.std() * (252 ** 0.5)
)

market_annual_volatility = (
    market_oos.std() * (252 ** 0.5)
)

print("\nOut-of-sample annualised metrics")
print(f"Strategy annualised return: {strategy_annual_return:.2%}")
print(f"Buy-and-hold annualised return: {market_annual_return:.2%}")
print(f"Strategy annualised volatility: {strategy_annual_volatility:.2%}")
print(f"Buy-and-hold annualised volatility: {market_annual_volatility:.2%}")




# Calculate out-of-sample drawdowns
strategy_oos_drawdown = (
    strategy_oos_growth / strategy_oos_growth.cummax()
) - 1

market_oos_drawdown = (
    market_oos_growth / market_oos_growth.cummax()
) - 1

# Find the largest peak-to-trough decline
strategy_oos_max_drawdown = strategy_oos_drawdown.min()
market_oos_max_drawdown = market_oos_drawdown.min()

print("\nOut-of-sample maximum drawdown")
print(f"Strategy: {strategy_oos_max_drawdown:.2%}")
print(f"Buy and hold: {market_oos_max_drawdown:.2%}")



# Calculate out-of-sample Sharpe ratios
strategy_oos_sharpe = (
    strategy_oos.mean() / strategy_oos.std()
) * (252 ** 0.5)

market_oos_sharpe = (
    market_oos.mean() / market_oos.std()
) * (252 ** 0.5)

print("\nOut-of-sample Sharpe ratios (zero risk-free rate)")
print(f"Strategy: {strategy_oos_sharpe:.2f}")
print(f"Buy and hold: {market_oos_sharpe:.2f}")




import matplotlib.pyplot as plt

# Calculate drawdown series
strategy_drawdown = (
    strategy_oos_growth / strategy_oos_growth.cummax()
) - 1

market_drawdown = (
    market_oos_growth / market_oos_growth.cummax()
) - 1

# Plot drawdowns
plt.figure(figsize=(12, 6))

plt.plot(
    strategy_drawdown.index,
    strategy_drawdown,
    label="SMA 20/50 Strategy"
)

plt.plot(
    market_drawdown.index,
    market_drawdown,
    label="Buy and Hold"
)

plt.title("Out-of-Sample Drawdown Comparison")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/oos_drawdown.png", dpi=300)
plt.show()



# Inspect the strategy's market exposure
position_oos = position.loc[strategy_oos.index]

cash_days = (position_oos == 0).sum()
invested_days = (position_oos == 1).sum()
total_days = len(position_oos)

print("\nOut-of-sample market exposure")
print(f"Days in cash: {cash_days}")
print(f"Days invested: {invested_days}")
print(f"Percentage of days in cash: {cash_days / total_days:.2%}")
print(f"Percentage of days invested: {invested_days / total_days:.2%}")



# Compare daily returns with market exposure
exposure_check = pd.DataFrame({
    "Position": position_oos,
    "Strategy_Return": strategy_oos,
    "Drawdown": strategy_drawdown
})

# Show observations where the strategy is in cash
cash_observations = exposure_check[
    exposure_check["Position"] == 0
]

print("\nExample observations while in cash")
print(cash_observations.head(15))

# Check whether returns are zero while in cash
print("\nAverage absolute strategy return while in cash:")
print(cash_observations["Strategy_Return"].abs().mean())



# Count position changes during the out-of-sample period
position_changes_oos = position_oos.diff()

entries = (position_changes_oos == 1).sum()
exits = (position_changes_oos == -1).sum()

print("\nOut-of-sample trade activity")
print(f"Entries (cash to invested): {int(entries)}")
print(f"Exits (invested to cash): {int(exits)}")
print(f"Total position changes: {int(entries + exits)}")



# Measure the length of each holding period
position_groups = position_oos.ne(position_oos.shift()).cumsum()

holding_periods = (
    position_oos[position_oos == 1]
    .groupby(position_groups[position_oos == 1])
    .size()
)

print("\nOut-of-sample holding periods")

if not holding_periods.empty:
    print(f"Number of holding periods: {len(holding_periods)}")
    print(f"Average holding period: {holding_periods.mean():.1f} trading days")
    print(f"Median holding period: {holding_periods.median():.1f} trading days")
    print(f"Shortest holding period: {holding_periods.min()} trading days")
    print(f"Longest holding period: {holding_periods.max()} trading days")
else:
    print("No holding periods found.")



# Calculate the return during each holding period
period_returns = {}

for group in holding_periods.index:
    period_mask = position_groups == group
    daily_period_returns = strategy_oos[period_mask]

    period_return = (1 + daily_period_returns.fillna(0)).prod() - 1
    period_returns[group] = period_return

period_returns = pd.Series(period_returns)

print("\nHolding-period performance")
print(f"Profitable periods: {(period_returns > 0).sum()}")
print(f"Losing periods: {(period_returns < 0).sum()}")
print(f"Average period return: {period_returns.mean():.2%}")
print(f"Best period return: {period_returns.max():.2%}")
print(f"Worst period return: {period_returns.min():.2%}")



# Identify entries and exits in the out-of-sample period
position_changes = position_oos.diff()

entry_dates = position_oos.index[position_changes == 1]
exit_dates = position_oos.index[position_changes == -1]

print("\nTrade ledger preparation")
print(f"Entry dates found: {len(entry_dates)}")
print(f"Exit dates found: {len(exit_dates)}")

print("\nFirst five entries:")
print(entry_dates[:5])

print("\nFirst five exits:")
print(exit_dates[:5])





# Pair each entry with the next exit after it
trades = []

for entry_date in entry_dates:
    later_exits = exit_dates[exit_dates > entry_date]

    if len(later_exits) > 0:
        exit_date = later_exits[0]

        # Include the entry and exit dates
        trade_dates = position_oos.loc[entry_date:exit_date].index

        # Calculate the compounded return across the trade
        trade_daily_returns = strategy_oos.loc[trade_dates]

        trade_return = (
            (1 + trade_daily_returns.fillna(0)).prod() - 1
        )

        trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": exit_date,
            "Holding_Days": len(trade_dates),
            "Trade_Return": trade_return
        })

trade_ledger = pd.DataFrame(trades)

print("\nCompleted trade ledger")
print(trade_ledger.to_string(index=False))

if not trade_ledger.empty:
    print("\nTrade return summary")
    print(f"Profitable trades: {(trade_ledger['Trade_Return'] > 0).sum()}")
    print(f"Losing trades: {(trade_ledger['Trade_Return'] < 0).sum()}")
    print(f"Average trade return: {trade_ledger['Trade_Return'].mean():.2%}")
    print(f"Best trade: {trade_ledger['Trade_Return'].max():.2%}")
    print(f"Worst trade: {trade_ledger['Trade_Return'].min():.2%}")




# Separate profitable and losing trades
winning_trades = trade_ledger.loc[
    trade_ledger["Trade_Return"] > 0,
    "Trade_Return"
]

losing_trades = trade_ledger.loc[
    trade_ledger["Trade_Return"] < 0,
    "Trade_Return"
]

# Calculate average winning and losing trades
average_winner = winning_trades.mean()
average_loser = losing_trades.mean()

# Calculate gross profits and gross losses
gross_profits = winning_trades.sum()
gross_losses = abs(losing_trades.sum())

# Calculate profit factor
profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")

# Calculate expectancy per completed trade
expectancy = trade_ledger["Trade_Return"].mean()

print("\nTrade Economics")
print(f"Average winning trade: {average_winner:.2%}")
print(f"Average losing trade: {average_loser:.2%}")
print(f"Gross profits: {gross_profits:.2%}")
print(f"Gross losses: {gross_losses:.2%}")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Expectancy per trade: {expectancy:.2%}")


# Calculate monthly returns for the out-of-sample period

strategy_monthly = (
    (1 + strategy_oos.fillna(0))
    .resample("ME")
    .prod() - 1
)

market_monthly = (
    (1 + market_oos.fillna(0))
    .resample("ME")
    .prod() - 1
)

# Combine both sets of monthly returns
monthly_returns = pd.DataFrame({
    "Strategy": strategy_monthly,
    "Buy_and_Hold": market_monthly
})

print("\nMonthly Returns: Latest 12 Months")
print(
    monthly_returns.tail(12).to_string(
        float_format=lambda x: f"{x:.2%}"
    )
)



import numpy as np
import matplotlib.pyplot as plt

# Arrange monthly strategy returns into a Year x Month table
strategy_heatmap = (
    strategy_monthly.mul(100)
    .to_frame("Return")
    .assign(
        Year=lambda df: df.index.year,
        Month=lambda df: df.index.month
    )
    .pivot(index="Year", columns="Month", values="Return")
    .reindex(columns=range(1, 13))
)

# Create the heatmap
fig, ax = plt.subplots(figsize=(13, 6))

values = strategy_heatmap.to_numpy()
limit = np.nanmax(np.abs(values))

chart = ax.imshow(
    np.ma.masked_invalid(values),
    aspect="auto",
    cmap="RdYlGn",
    vmin=-limit,
    vmax=limit
)

# Label the months and years
months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

ax.set_xticks(range(12))
ax.set_xticklabels(months)
ax.set_yticks(range(len(strategy_heatmap.index)))
ax.set_yticklabels(strategy_heatmap.index)

# Print each monthly return inside its cell
for row in range(values.shape[0]):
    for col in range(values.shape[1]):
        value = values[row, col]

        if not np.isnan(value):
            ax.text(
                col, row, f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8
            )

ax.set_title("SMA 20/50 Strategy: Monthly Returns")
ax.set_xlabel("Month")
ax.set_ylabel("Year")

fig.colorbar(chart, ax=ax, label="Monthly return (%)")
plt.tight_layout()

plt.savefig("results/monthly_returns_heatmap.png", dpi=300)
plt.show()





# Compare monthly strategy returns with buy-and-hold

monthly_comparison = monthly_returns * 100

ax = monthly_comparison.plot(
    kind="bar",
    figsize=(14, 6),
    width=0.85
)

ax.set_title("Monthly Returns: SMA Strategy vs Buy and Hold")
ax.set_xlabel("Month")
ax.set_ylabel("Return (%)")
ax.axhline(0, linewidth=0.8)
ax.legend(["SMA 20/50 Strategy", "Buy and Hold"])
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/monthly_returns_comparison.png", dpi=300)
plt.show()




from pathlib import Path

# Make sure the results folder exists
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. EXPORT THE OUT-OF-SAMPLE PERFORMANCE REPORT
# --------------------------------------------------

performance_report = pd.DataFrame({
    "Metric": [
        "Total Return",
        "Annualised Return (CAGR)",
        "Annualised Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown"
    ],
    "Strategy": [
        strategy_oos_return,
        strategy_annual_return,
        strategy_annual_volatility,
        strategy_oos_sharpe,
        strategy_oos_max_drawdown
    ],
    "Buy_and_Hold": [
        market_oos_return,
        market_annual_return,
        market_annual_volatility,
        market_oos_sharpe,
        market_oos_max_drawdown
    ]
})

performance_report.to_csv(
    results_folder / "performance_report.csv",
    index=False
)


# --------------------------------------------------
# 2. EXPORT THE COMPLETED TRADE LEDGER
# --------------------------------------------------

trade_ledger_export = trade_ledger.copy()

# Remove timezone information from dates for easier Excel use
for column in ["Entry_Date", "Exit_Date"]:
    if column in trade_ledger_export.columns:
        trade_ledger_export[column] = (
            pd.to_datetime(trade_ledger_export[column])
            .dt.tz_localize(None)
        )

trade_ledger_export.to_csv(
    results_folder / "trade_ledger.csv",
    index=False
)


# --------------------------------------------------
# 3. EXPORT THE TRADE STATISTICS
# --------------------------------------------------

number_of_trades = len(trade_ledger)

number_of_winners = len(winning_trades)
number_of_losers = len(losing_trades)

win_rate = (
    number_of_winners / number_of_trades
    if number_of_trades > 0
    else float("nan")
)

trade_statistics = pd.DataFrame({
    "Metric": [
        "Completed Trades",
        "Winning Trades",
        "Losing Trades",
        "Win Rate",
        "Average Trade Return",
        "Average Winning Trade",
        "Average Losing Trade",
        "Sum of Winning Trade Returns",
        "Sum of Absolute Losing Trade Returns",
        "Profit Factor",
        "Expectancy per Trade"
    ],
    "Value": [
        number_of_trades,
        number_of_winners,
        number_of_losers,
        win_rate,
        expectancy,
        average_winner,
        average_loser,
        gross_profits,
        gross_losses,
        profit_factor,
        expectancy
    ]
})

trade_statistics.to_csv(
    results_folder / "trade_statistics.csv",
    index=False
)


# --------------------------------------------------
# 4. CONFIRM THAT THE REPORTS WERE SAVED
# --------------------------------------------------

print("\nCSV reports exported successfully!")

print(f"Performance report: {results_folder / 'performance_report.csv'}")
print(f"Trade ledger: {results_folder / 'trade_ledger.csv'}")
print(f"Trade statistics: {results_folder / 'trade_statistics.csv'}")
