import os
import gym
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from HybridEnv import HybridEnv

# === Параметры ===
ticker = "GAZP"
data_path = f"../../Data/TQBR.{ticker}_M5.txt"
start_date = "01.01.2019"
end_date = "31.12.2024"
window_size = 60
commission = 0.0005
total_timesteps = 600_000
log_path = "./ppo_lstm_hybrid_logs"
checkpoint_path = "./ppo_lstm_hybrid_checkpoints"
model_path = "./ppo_lstm_hybrid_model"

# === Логирование в файл ===
log_filename = os.path.join(log_path, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
os.makedirs(log_path, exist_ok=True)
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

# === Загрузка данных ===
df = pd.read_csv(data_path, sep="\t")
df.columns = df.columns.str.lower()
df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M", dayfirst=True)
df.set_index("datetime", inplace=True)
df = df[start_date:end_date].replace([np.inf, -np.inf], np.nan).dropna().reset_index()

# === Среда ===
env = DummyVecEnv([lambda: HybridEnv(df.copy(), window_size=window_size, commission=commission)])
# env = VecNormalize(env, norm_obs=False, norm_reward=True)

# === Кастомный колбэк для сохранения модели и нормализации ===
class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="model", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            model_file = os.path.join(self.save_path, f"{self.name_prefix}_{step}_steps.zip")
            #vec_file = os.path.join(self.save_path, f"{self.name_prefix}_{step}_vecnormalize.pkl")
            self.model.save(model_file)

            #vec_env = self.training_env
            #if isinstance(vec_env, VecNormalize):
            #    vec_env.save(vec_file)
            #elif hasattr(vec_env, "venv") and isinstance(vec_env.venv, VecNormalize):
            #    vec_env.venv.save(vec_file)

            #if self.verbose:
            #    print(f"Сохранён чекпоинт: {model_file} и нормализация: {vec_file}")

        return True

# === Параметры политики ===
policy_kwargs = dict(
    net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
    lstm_hidden_size=256,
    n_lstm_layers=1,
    ortho_init=False,
    activation_fn=torch.nn.ReLU,
)

model = RecurrentPPO(
    policy=MlpLstmPolicy,
    env=env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=256,
    clip_range=0.1,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.98,
    ent_coef=0.01,
    normalize_advantage=True,
    tensorboard_log=log_path,
    policy_kwargs=policy_kwargs,
)

model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))

checkpoint_callback = CustomCheckpointCallback(
    save_freq=25_000,
    save_path=checkpoint_path,
    name_prefix="ppo_lstm_hybrid_checkpoint",
    verbose=1
)

try:
    print(f"Обучение начато: {datetime.now()}")

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
    model.save(model_path)
    #env.save("vecnormalize.pkl")

    print(f"Финальная модель сохранена в: {model_path}")

except Exception as e:
    print("Ошибка во время обучения:", str(e))
    for name, param in model.policy.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN в параметре: {name}")
        if torch.isinf(param).any():
            print(f"Inf в параметре: {name}")
