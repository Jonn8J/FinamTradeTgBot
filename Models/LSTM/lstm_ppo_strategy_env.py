import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from sklearn.preprocessing import StandardScaler

class LSTMPPOStrategy:
    def __init__(self, model_path, window_size=60):
        self.model = RecurrentPPO.load(model_path)
        self.window_size = window_size

        self.features = [
            "open", "high", "low", "close", "volume",
            "log_return", "candle_body", "upper_shadow", "lower_shadow"
        ]
        self.scaler = StandardScaler()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["candle_body"] = df["close"] - df["open"]
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

        df.dropna(inplace=True)
        df[self.features] = self.scaler.fit_transform(df[self.features])
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        original_index = df.index
        df = df.copy().sort_index()
        df = self._prepare_features(df)

        obs = df[self.features].values.astype(np.float32)
        signals = ["hold"] * self.window_size

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        for i in range(self.window_size, len(obs)):
            seq = obs[i - self.window_size:i]
            seq = np.expand_dims(seq, axis=0)  # (1, window_size, num_features)

            action, lstm_states = self.model.predict(
                seq,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )

            signal = {0: "hold", 1: "buy", 2: "sell"}.get(int(action), "hold")
            signals.append(signal)
            episode_starts = np.array([False])  # not a new episode anymore

        # Вернём серию сигналов, выровненную по индексу
        full_signals = pd.Series("hold", index=original_index)
        full_signals.iloc[-len(signals):] = signals
        return full_signals