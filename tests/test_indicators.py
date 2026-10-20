
def test_basic_arithmetic():
    price = 100
    new_price = 110

    daily_return = (new_price - price) / price

    assert daily_return == 0.10




import pandas as pd


def test_basic_arithmetic():
    price = 100
    new_price = 110

    daily_return = (new_price - price) / price

    assert daily_return == 0.10


def test_daily_returns():
    prices = pd.Series([100, 110, 99])

    returns = prices.pct_change()

    assert pd.isna(returns.iloc[0])
    assert abs(returns.iloc[1] - 0.10) < 1e-10
    assert abs(returns.iloc[2] - (-0.10)) < 1e-10



def test_moving_averages():
    prices = pd.Series([10, 20, 30, 40, 50])

    sma_3 = prices.rolling(window=3).mean()

    assert pd.isna(sma_3.iloc[0])
    assert pd.isna(sma_3.iloc[1])
    assert sma_3.iloc[2] == 20
    assert sma_3.iloc[3] == 30
    assert sma_3.iloc[4] == 40




def test_trading_signals():
    sma_20 = pd.Series([101, 99, 100, 105])
    sma_50 = pd.Series([100, 100, 100, 103])

    signal = (sma_20 > sma_50).astype(float)

    assert signal.tolist() == [1.0, 0.0, 0.0, 1.0]




def test_position_is_delayed_by_one_day():
    signal = pd.Series([1.0, 0.0, 1.0, 1.0])

    position = signal.shift(1)

    assert pd.isna(position.iloc[0])
    assert position.iloc[1] == 1.0
    assert position.iloc[2] == 0.0
    assert position.iloc[3] == 1.0
