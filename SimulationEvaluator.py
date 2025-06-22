import os
import pandas as pd
import numpy as np
from Models.strategy_registry import get_strategy
import re
import time


# === Параметры ===
model_name = "Hybrid"
ticker = "GAZP"
data_file = f"Data/TQBR.{ticker}_M5.txt"
start_date = "01.12.2024"
end_date = "05.04.2025"
commission = 0.0005
initial_balance = 100000
checkpoints_dir = "Models/Hybrid/ppo_lstm_hybrid_checkpoints"
#checkpoints_dir = "Models/LSTM/LSTM_PPO/ppo_lstm_checkpoints"

# === Загрузка данных ===
df = pd.read_csv(data_file, sep="\t")
df.columns = df.columns.str.lower()
df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M")
df = df.set_index("datetime")
df = df[(df.index >= pd.to_datetime(start_date, format="%d.%m.%Y")) &
        (df.index <= pd.to_datetime(end_date, format="%d.%m.%Y"))]
if isinstance(df.index, pd.DatetimeIndex):
    df["datetime"] = df.index
#    match = re.search(r"ppo_lstm_checkpoint_(\d+)_steps\.zip", filename)
# match = re.search(r"ppo_lstm_hybrid_checkpoint_(\d+)_steps\.zip", filename)
def extract_steps(filename):
    match = re.search(r"ppo_lstm_hybrid_checkpoint_(\d+)_steps\.zip", filename)
    return int(match.group(1)) if match else float('inf')

# === Поиск checkpoint-файлов ===
checkpoints = [
    f for f in os.listdir(checkpoints_dir)
    if f.endswith(".zip") and f.startswith("ppo_lstm_hybrid_checkpoint_")
]
# if f.endswith(".zip") and f.startswith("ppo_lstm_hybrid_checkpoint_")
checkpoints = sorted(checkpoints, key=extract_steps)

results = []

for chk in checkpoints:
    chk_path = os.path.join(checkpoints_dir, chk)
    #vecnorm_path = chk_path.replace("_steps.zip", "_vecnormalize.pkl")

    print(f"Оценка: {chk}")
    #if not os.path.exists(vecnorm_path):
    #    print(f"Пропущено: {chk} — vecnormalize файл не найден: {vecnorm_path}")
    #    continue

    try:
        strategy = get_strategy(model_name, model_path=chk_path)
        signals = strategy.generate_signals(df)
    except Exception as e:
        print(f"Пропущено: {chk} — ошибка при генерации сигналов: {e}")
        continue

    balance = initial_balance
    quantity = 0
    entry_price = 0
    in_position = False
    balance_trace = []

    for time, row in df.iterrows():
        if time not in signals.index:
            continue

        signal = signals.loc[time]
        price = row['close']

        if signal == 'buy' and not in_position:
            entry_price = price
            quantity = balance // entry_price
            balance -= quantity * entry_price
            in_position = True

        elif signal == 'sell' and in_position:
            balance += quantity * price * (1 - commission)
            quantity = 0
            in_position = False

        equity = balance + quantity * price
        balance_trace.append(equity)

    if not balance_trace:
        print(f"Нет данных по сделкам: {chk}")
        continue

    balance_series = pd.Series(balance_trace, index=df.index[:len(balance_trace)])
    returns = balance_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty else 0
    max_dd = ((balance_series.cummax() - balance_series) / balance_series.cummax()).max()

    results.append({
        "checkpoint": chk,
        "final_balance": round(balance_series.iloc[-1], 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd * 100, 2),
        "max_equity": round(balance_series.max(), 2),
        "min_equity": round(balance_series.min(), 2)
    })

# === Сохранение результатов ===
os.makedirs("Simulate_data/Evaluator", exist_ok=True)
pd.DataFrame(results).sort_values("final_balance", ascending=False).to_csv(
    "Simulate_data/Evaluator/checkpoint_summary.csv", index=False
)
print("Сводная таблица сохранена в Simulate_data/Evaluator/checkpoint_summary.csv")