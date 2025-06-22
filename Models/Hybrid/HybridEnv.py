import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

class HybridEnv(gym.Env):
    def __init__(self, df, window_size=60, commission=0.0005):
        super().__init__()
        self.df = df.copy()

        # === Индикаторы тренда и моментума ===
        self.df['sma_10'] = SMAIndicator(self.df['close'], window=10).sma_indicator()
        self.df['sma_50'] = SMAIndicator(self.df['close'], window=50).sma_indicator()
        self.df['rsi'] = RSIIndicator(self.df['close'], window=14).rsi()
        macd = MACD(self.df['close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        stoch = StochasticOscillator(self.df['high'], self.df['low'], self.df['close'])
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()

        # === Волатильность ===
        atr = AverageTrueRange(self.df['high'], self.df['low'], self.df['close'], window=14)
        self.df['atr'] = atr.average_true_range()

        bb = BollingerBands(self.df['close'], window=20, window_dev=2)
        self.df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()

        # === Объём ===
        self.df['volume_sma_20'] = self.df['volume'].rolling(window=20).mean()
        self.df['obv'] = OnBalanceVolumeIndicator(close=self.df['close'], volume=self.df['volume']).on_balance_volume()
        self.df['volume_price_ratio'] = self.df['volume'] / (self.df['close'] + 1e-6)

        # === Price action ===
        self.df["log_return"] = np.log(self.df["close"] / self.df["close"].shift(1))
        self.df["candle_body"] = self.df["close"] - self.df["open"]
        self.df["upper_shadow"] = self.df["high"] - self.df[["open", "close"]].max(axis=1)
        self.df["lower_shadow"] = self.df[["open", "close"]].min(axis=1) - self.df["low"]
        self.df["volume_change"] = self.df["volume"].pct_change()

        # === Временные признаки ===
        self.df["hour"] = self.df["datetime"].dt.hour
        self.df["hour_sin"] = np.sin(2 * np.pi * self.df["hour"] / 24)
        self.df["hour_cos"] = np.cos(2 * np.pi * self.df["hour"] / 24)
        self.df.drop(columns=["datetime", "hour"], inplace=True)

        # === Очистка ===
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("NaN counts before dropna:\n", self.df.isna().sum())
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # === Стандартизация (кроме цены) ===
        self.window_size = window_size
        self.commission = commission
        self.initial_balance = 100000
        self.df["close_raw"] = self.df["close"]
        self.features = [col for col in self.df.columns if col != "close_raw"]

        self.scaler = StandardScaler()
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.features)),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.step_idx = self.window_size
        self.entry_price = 0
        self.quantity = 0
        self.in_position = False
        return self._get_observation()

    def _get_price(self):
        return self.df.iloc[self.step_idx]["close_raw"]

    def _get_observation(self):
        obs = self.df[self.features].iloc[self.step_idx - self.window_size:self.step_idx].values
        if np.isnan(obs).any():
            print(f"NaN в OBS на step_idx={self.step_idx}")
            print(obs)
        return obs.astype(np.float32)

    def step(self, action):
        price = self._get_price()
        reward = 0.0

        if price <= 0 or np.isnan(price):
            print(f"Некорректная цена: {price} на step={self.step_idx}")

        if action == 1 and not self.in_position:
            self.entry_price = price
            self.quantity = self.initial_balance // self.entry_price
            self.in_position = True

        elif action == 2 and self.in_position:
            profit = ((price - self.entry_price) * self.quantity * (1 - self.commission)) / self.initial_balance
            reward = profit
            self.quantity = 0
            self.entry_price = 0
            self.in_position = False

        if np.isnan(reward) or np.isinf(reward):
            print(f"reward=NaN | step={self.step_idx} | price={price} | entry_price={self.entry_price} | qty={self.quantity} | reward={reward}")

        self.step_idx += 1
        self.done = self.step_idx >= len(self.df) - 1

        return self._get_observation(), reward, self.done, {
            "price": price,
            "entry_price": self.entry_price,
            "reward": reward,
        }
