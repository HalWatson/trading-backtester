
import yfinance as yf

# Download daily QQQ prices from 2010 onwards.
ticker = yf.Ticker("QQQ")

data = ticker.history(
    start="2010-01-01",
    interval="1d",
    auto_adjust=True
)

# Inspect the downloaded data.
print(data.head())
print(data.tail())
print("Number of rows:", len(data))
print("Columns:", data.columns.tolist())
data.to_csv("data/qqq_daily.csv")

