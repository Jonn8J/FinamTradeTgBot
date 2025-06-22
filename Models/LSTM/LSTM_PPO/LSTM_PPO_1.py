from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

import pandas as pd
import numpy as np
import torch
import os
import sys
from datetime import datetime
from Models.LSTM.LSTM_PPO.LSTMEnv import CustomLSTMEnv

# === Логирование в файл ===
log_filename = os.path.join("./ppo_lstm_logs", f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
os.makedirs("./ppo_lstm_logs", exist_ok=True)
log_file = open(log_filename, "w", encoding="utf-8")

class Logger(object):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Logger(sys.stdout, log_file)
sys.stderr = Logger(sys.stderr, log_file)

# === Ограничения на потоки CPU ===
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)

# === Колбэк: Градиентный клиппинг ===
class GradientClippingCallback(BaseCallback):
    def __init__(self, clip_value=0.5, verbose=0):
        super().__init__(verbose)
        self.clip_value = clip_value
    def _on_step(self) -> bool:
        for param in self.model.policy.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, self.clip_value)
        return True

# === Параметры ===
ticker = "GAZP"
data_path = f"../../../Data/TQBR.{ticker}_M5.txt"
model_save_path = f"ppo_lstm_{ticker.lower()}_final_env2_model"
log_path = "./ppo_lstm_logs"
checkpoint_path = "./ppo_lstm_checkpoints"
start_date = "01.01.2019"
end_date = "31.12.2024"
window_size = 60
commission = 0.0005
total_timesteps = 500_000

# === Загрузка данных ===
df = pd.read_csv(data_path, sep="\t")
df.columns = df.columns.str.lower()
df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M", dayfirst=True)
df.set_index("datetime", inplace=True)
df = df[start_date:end_date].replace([np.inf, -np.inf], np.nan).dropna().reset_index()

# === Среда ===
env = DummyVecEnv([lambda: CustomLSTMEnv(df=df, window_size=window_size, commission=commission)])

# === Логгер ===
os.makedirs(log_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)
new_logger = configure(log_path, ["stdout", "csv", "tensorboard", "json"])

# === PPO-LSTM модель ===
model = RecurrentPPO(
    policy=MlpLstmPolicy,
    env=env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.001,
    tensorboard_log=log_path,
    normalize_advantage=True,
    policy_kwargs={
        "n_lstm_layers": 2,
        "lstm_hidden_size": 128,
        "activation_fn": torch.nn.ReLU,
        "net_arch": dict(pi=[256, 128, 64], vf=[256, 128, 64])
    }
)
model.set_logger(new_logger)

# === Колбэки ===
callbacks = CallbackList([
    GradientClippingCallback(clip_value=0.5),
    CheckpointCallback(save_freq=25_000, save_path=checkpoint_path, name_prefix="ppo_lstm_checkpoint")
])

# === Обучение ===
try:
    print(f"Обучение начато: {datetime.now()}")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(model_save_path)
    print(f"Финальная модель сохранена: {model_save_path} | {datetime.now()}")

except KeyboardInterrupt:
    print("Обучение остановлено вручную.")
except Exception as e:
    print(f"Ошибка: {str(e)}")
