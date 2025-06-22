import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, MACD

class LSTMPPOHybridStrategy:
    def __init__(self, model_path, window_size=60):
        self.model = RecurrentPPO.load(model_path)
        self.window_size = window_size

        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_10', 'sma_50', 'rsi',
            'macd', 'macd_signal',
            'stoch_k', 'stoch_d', 'volume_change',
            'hour_sin', 'hour_cos'
        ]
        self.scaler = StandardScaler()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df.reset_index()

        df['sma_10'] = SMAIndicator(df['close'], window=10).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        df['volume_change'] = df['volume'].pct_change()
        df["hour"] = df["datetime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df.drop(columns=["datetime", "hour"], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df[self.features] = self.scaler.fit_transform(df[self.features])

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        original_index = df.index
        df = df.copy().sort_index()
        df = self._prepare_features(df)
        df = df.reset_index(drop=False)

        full_features = self.features
        obs = df[full_features].values.astype(np.float32)

        signals = ["hold"] * self.window_size
        lstm_states = None
        episode_start_flags = [True] + [False] * (len(obs) - self.window_size - 1)

        for i in range(self.window_size, len(obs)):
            seq = obs[i - self.window_size:i]
            seq = np.expand_dims(seq, axis=0)

            action, lstm_states = self.model.predict(
                seq,
                state=lstm_states,
                episode_start=np.array([episode_start_flags[i - self.window_size]]),
                deterministic=True
            )

            signal = {0: "hold", 1: "buy", 2: "sell"}.get(int(action), "hold")
            signals.append(signal)

        full_signals = pd.Series(["hold"] * len(original_index), index=original_index)
        full_signals.iloc[-len(signals):] = signals
        return full_signals
