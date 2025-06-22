import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler

class CustomLSTMEnv(gym.Env):
    def __init__(self, df, window_size=60, commission=0.0005):
        super().__init__()
        self.df = df.copy()
        self.df = self.df.dropna().reset_index().drop(columns=["datetime"])
        self.window_size = window_size
        self.commission = commission
        self.initial_balance = 100_000

        self.df["log_return"] = np.log(self.df["close"] / self.df["close"].shift(1))
        self.df["candle_body"] = self.df["close"] - self.df["open"]
        self.df["upper_shadow"] = self.df["high"] - self.df[["open", "close"]].max(axis=1)
        self.df["lower_shadow"] = self.df[["open", "close"]].min(axis=1) - self.df["low"]
        self.df = self.df.dropna().reset_index()

        self.features = [
            "open", "high", "low", "close", "volume",
            "log_return", "candle_body", "upper_shadow", "lower_shadow"
        ]

        self.scaler = StandardScaler()
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

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
        return self.df.iloc[self.step_idx]["close"]

    def _get_observation(self):
        obs = self.df[self.features].iloc[self.step_idx - self.window_size:self.step_idx].values
        return obs.astype(np.float32)

    def step(self, action):
        price = self._get_price()
        reward = 0.0

        if action == 1 and not self.in_position:  # BUY
            self.entry_price = price
            self.quantity = self.initial_balance // self.entry_price
            self.in_position = True

        elif action == 2 and self.in_position:  # SELL
            profit = ((price - self.entry_price) * self.quantity * (1 - self.commission)) / self.initial_balance
            reward = profit
            self.quantity = 0
            self.entry_price = 0
            self.in_position = False

        self.step_idx += 1
        self.done = self.step_idx >= len(self.df) - 1

        return self._get_observation(), reward, self.done, {
            "price": price,
            "entry_price": self.entry_price,
            "reward": reward,
        }
