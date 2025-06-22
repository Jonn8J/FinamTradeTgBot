import pandas as pd
from Models.Base.base_strategy import BaseStrategy

# MACD Strategy (простая логика)
class MACDStrategy:
    def generate_signals(self, df):
        df = df.copy()
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['position'] = 0
        df.loc[df['macd'] > df['signal'], 'position'] = 1
        df.loc[df['macd'] < df['signal'], 'position'] = -1
        return df['position'].replace({1: 'buy', -1: 'sell', 0: 'hold'})

class SMAStrategy(BaseStrategy):
    def __init__(self, fast_period=10, slow_period=50):
        self.fast = fast_period
        self.slow = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df["sma_fast"] = df["close"].rolling(self.fast).mean()
        df["sma_slow"] = df["close"].rolling(self.slow).mean()
        df.dropna(inplace=True)

        signal = pd.Series("hold", index=df.index)
        signal[df["sma_fast"] > df["sma_slow"]] = "buy"
        signal[df["sma_fast"] < df["sma_slow"]] = "sell"
        return signal
