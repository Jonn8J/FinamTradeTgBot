import numpy as np
import pandas as pd
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from Models.Hybrid.HybridEnv import HybridEnv

class LSTMPPOHybridStrategy2:
    def __init__(self, model_path, window_size=60):
        self.model_path = model_path
       # self.vecnorm_path = model_path.replace("_steps.zip", "_vecnormalize.pkl")
        #self.vecnorm_path = vecnorm_path
        self.window_size = window_size

        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_10', 'sma_50', 'rsi',
            'macd', 'macd_signal',
            'stoch_k', 'stoch_d',
            'atr', 'bb_width',
            'volume_sma_20', 'obv', 'volume_price_ratio',
            'log_return', 'candle_body', 'upper_shadow', 'lower_shadow', 'volume_change',
            'hour_sin', 'hour_cos'
        ]
        self.scaler = StandardScaler()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = df.index

        # === Индикаторы тренда и моментума ===
        df['sma_10'] = SMAIndicator(df['close'], window=10).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # === Волатильность ===
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()

        # === Объём ===
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['volume_price_ratio'] = df['volume'] / (df['close'] + 1e-6)

        # === Price action ===
        df["log_return"] = np.log(df["close"].replace(0, np.nan) / df["close"].shift(1)).replace([np.inf, -np.inf], np.nan)
        df["candle_body"] = df["close"] - df["open"]
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["volume_change"] = df["volume"].pct_change()

        # === Время ===
        df["hour"] = df["datetime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        # df.drop(columns=["datetime", "hour"], inplace=True)
        df.drop(columns=["hour"], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df[self.features] = self.scaler.fit_transform(df[self.features])
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        original_index = df.index
        df = df.copy().sort_index()
        df = self._prepare_features(df)

        # Временный VecEnv для подстановки
        #dummy_env = DummyVecEnv([lambda: HybridEnv(df.copy(), window_size=self.window_size)])
        #if os.path.exists(self.vecnorm_path):
        #    env = VecNormalize.load(self.vecnorm_path, dummy_env)
        #    env.training = False
        #    env.norm_reward = False
        #else:
        #    print(f"⚠️ VecNormalize не найден: {self.vecnorm_path}")
        #    env = dummy_env

        model = RecurrentPPO.load(self.model_path)

        obs = df[self.features].values.astype(np.float32)
        signals = ["hold"] * self.window_size
        lstm_states = None
        episode_start_flags = [True] + [False] * (len(obs) - self.window_size - 1)

        for i in range(self.window_size, len(obs)):
            seq = obs[i - self.window_size:i]
            seq = np.expand_dims(seq, axis=0)

            #if isinstance(env, VecNormalize):
            #   seq = env.normalize_obs(seq)

            action, lstm_states = model.predict(
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
