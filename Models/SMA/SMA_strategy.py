import pandas as pd
from Models.Base.base_strategy import BaseStrategy

class SMAStrategy(BaseStrategy):
    def __init__(self, fast_period=15, slow_period=50):
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
