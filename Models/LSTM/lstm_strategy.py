import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from Models.Base.base_strategy import BaseStrategy
from sklearn.preprocessing import MinMaxScaler

class LSTMStrategy(BaseStrategy):
    def __init__(self, model_path: str, sequence_length: int = 50):
        self.model = load_model(model_path)
        self.seq_len = sequence_length
        self.scaler = MinMaxScaler()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df = df[['open', 'high', 'low', 'close', 'volume']]

        data = self.scaler.fit_transform(df)
        X = []

        for i in range(self.seq_len, len(data)):
            X.append(data[i - self.seq_len:i])

        X = np.array(X)
        preds = self.model.predict(X, verbose=0)
        signals = []

        for prob in preds:
            cls = np.argmax(prob)
            if cls == 0:
                signals.append("sell")
            elif cls == 1:
                signals.append("hold")
            else:
                signals.append("buy")

        # Добавим "hold" в начало, т.к. предсказания начинаются после seq_len
        signals = ["hold"] * self.seq_len + signals
        return pd.Series(signals, index=df.index[:len(signals)])
