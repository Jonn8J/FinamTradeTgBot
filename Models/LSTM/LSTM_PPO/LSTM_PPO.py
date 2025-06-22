from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

from TradingEnvRl import TradingEnv

import pandas as pd
import numpy as np
import torch
import os

class GradientClippingCallback(BaseCallback):
    def __init__(self, clip_value=0.5, verbose=0):
        super().__init__(verbose)
        self.clip_value = clip_value

    def _on_step(self) -> bool:
        for param in self.model.policy.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, self.clip_value)
        return True

# === Parameters ===
ticker = "GAZP"
data_path = f"../../Data/TQBR.{ticker}_M5.txt"
model_save_path = f"ppo_lstm_{ticker.lower()}_model"
log_path = "./ppo_lstm_logs"
start_date = "01.01.2022"
end_date = "31.12.2024"
window_size = 60
initial_balance = 100000
commission = 0.0005
total_timesteps = 300_000

# === Load Data ===
df = pd.read_csv(data_path, sep="\t")
df.columns = df.columns.str.lower()
df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M", dayfirst=True)
df.set_index("datetime", inplace=True)
df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# === Environment ===
env = DummyVecEnv([lambda: TradingEnv(df=df, window_size=window_size, start_balance=initial_balance, commission=commission)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1.0, clip_reward=1.0)

# === Logger ===
os.makedirs(log_path, exist_ok=True)
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

# === PPO + LSTM ===
model = RecurrentPPO(
    policy=MlpLstmPolicy,
    env=env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    tensorboard_log=log_path,
    policy_kwargs={
        "n_lstm_layers": 1,
        "lstm_hidden_size": 64,
        "activation_fn": torch.nn.Tanh,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
    }
)

model.set_logger(new_logger)
model.learn(total_timesteps=total_timesteps, callback=GradientClippingCallback(clip_value=0.5))
model.save(model_save_path)
print(f"✅ PPO-LSTM model saved to: {model_save_path}")