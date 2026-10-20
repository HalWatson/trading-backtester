

from pathlib import Path

import yfinance as yf

# Download daily, adjusted QQQ prices from 2010 onwards.
ticker = yf.Ticker("QQQ")

data = ticker.history(
    start="2010-01-01",
    interval="1d",
    auto_adjust=True,
)

if data.empty:
    raise RuntimeError("QQQ download returned no data. Dataset not updated.")

# Inspect the downloaded dataset.
print("\nFirst five rows:")
print(data.head())

print("\nLast five rows:")
print(data.tail())

print("\nDataset information:")
data.info()

print("\nMissing values per column:")
print(data.isna().sum())

print("\nDataset shape:", data.shape)

# Save the dataset relative to the project root.
output_path = Path(__file__).resolve().parent.parent / "data" / "qqq_daily.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(output_path)

print(f"\nSaved {len(data)} rows to {output_path}")
