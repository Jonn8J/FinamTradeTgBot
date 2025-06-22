import numpy as np
from sb3_contrib import RecurrentPPO
import pandas as pd

class LSTMPPOStrategy:
    def __init__(self, model_path):
        self.model = RecurrentPPO.load(model_path)
        self.lstm_states = None
        self.episode_start = True
        self.window_size = 60  # обязательно такой же, как при обучении

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy().sort_index()
        obs_columns = ['open', 'high', 'low', 'close', 'volume']
        obs = df[obs_columns].values.astype(np.float32)

        signals = []
        self.lstm_states = None
        self.episode_start = True

        for i in range(len(obs)):
            if i < self.window_size:
                signals.append("hold")
                continue

            # Получаем срез длиной window_size
            seq = obs[i - self.window_size:i]
            seq = np.expand_dims(seq, axis=0)  # (1, 60, 5)

            action, self.lstm_states = self.model.predict(
                seq,
                state=self.lstm_states,
                episode_start=np.array([self.episode_start]),
                deterministic=True
            )

            signal = {0: "hold", 1: "buy", 2: "sell"}.get(int(action), "hold")
            signals.append(signal)
            self.episode_start = False

        return pd.Series(signals, index=df.index)
