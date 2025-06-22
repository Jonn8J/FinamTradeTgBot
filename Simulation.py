import pandas as pd
import matplotlib.pyplot as plt
import asyncio
import os
from my_config.Cfg import Config
from Models.strategy_registry import get_strategy
from Bars_graph import plot_trades

# Симуляция стратегии
def backtest(df, strategy, start_balance=100000, commission=0.0005,
             ticker="TICKER", model_name="SMA",
             start_date_str="01.12.2024", end_date_str="05.04.2025",
             log_path="Simulate_data/stats_log.txt",
             plot_file="Simulate_data/backtest_balance.png"):


    signals = strategy.generate_signals(df)
    balance = start_balance
    quantity = 0
    entry_price = 0
    in_position = False
    balance_over_time = []
    trades = []

    max_price_since_entry = None

    for time, row in df.iterrows():
        price = row['close']
        signal = signals.loc[time] if time in signals.index else 'hold'

        if signal == 'buy' and not in_position:
            entry_price = price
            quantity = balance // entry_price
            balance -= quantity * entry_price
            in_position = True
            max_price_since_entry = price
            trades.append((time, 'BUY', price))

        elif in_position:
            max_price_since_entry = max(max_price_since_entry, price)

            if signal == 'sell':
                balance += quantity * price * (1 - commission)
                trades.append((time, 'SELL', price))
                quantity = 0
                in_position = False

        portfolio_value = balance + quantity * price
        balance_over_time.append((time, portfolio_value))

    balance_df = pd.DataFrame(balance_over_time, columns=['datetime', 'balance']).set_index('datetime')

    final_balance = balance_df['balance'].iloc[-1]
    returns = balance_df['balance'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if not returns.empty else 0
    max_drawdown = ((balance_df['balance'].cummax() - balance_df['balance']) / balance_df['balance'].cummax()).max()

    # Формируем текст отчета в красивом виде
    report_lines = [
        f"{'='*60}",
        f"📊 Результаты симуляции стратегии",
        f"🎯 Актив: {ticker}",
        f"⚙️ Стратегия: {model_name}",
        f"📆 Период: {start_date_str} — {end_date_str}",
        f"🏦 Начальный баланс: {start_balance} руб.",
        "",
        f"💰 Финальный баланс: {final_balance:.2f} руб.",
        f"📉 Максимальная просадка: {max_drawdown:.2%}",
        f"📈 Коэффициент Шарпа: {sharpe_ratio:.4f}",
        f"🔄 Совершено сделок: {len(trades)}",
        f"{'='*60}\n"
    ]

    # Печать в консоль
    print("\n".join(report_lines))

    # Сохраняем в единый файл (добавляем в конец)
    with open(log_path, 'a', encoding='utf-8') as f:
        for line in report_lines:
            f.write(line + '\n')

    print(f"Статистика добавлена в: {log_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(balance_df.index, balance_df['balance'], label='Баланс', linewidth=2)
    plt.title(f"График изменения баланса — {ticker}")
    plt.xlabel("Дата")
    plt.ylabel("Баланс, руб.")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)

    print(f"График баланса сохранён в: {plot_file}")

    return balance_df, trades


from datetime import time

async def generate_trade_graphs(trades, df, user_id, security_board, ticker, time_frame="M5", model=""):
    df = df.copy()
    df = df.sort_index()
    df["date"] = df.index.date
    df["time"] = df.index.time

    # Получаем последние два уникальных дня
    unique_dates = sorted(df["date"].unique())[-2:]

    if len(unique_dates) < 2:
        print("Недостаточно данных для двух дней.")
        return

    sessions = []

    for day in unique_dates:
        # Первая половина
        morning = df[(df["date"] == day) & (df["time"] >= time(7, 0)) & (df["time"] <= time(15, 25))]
        # Вторая половина
        evening = df[(df["date"] == day) & (df["time"] > time(15, 25)) & (df["time"] <= time(23, 50))]

        if not morning.empty:
            sessions.append(morning)
        if not evening.empty:
            sessions.append(evening)

    # Подготовка сделок
    parsed_trades = [
        {
            "timestamp": t,
            "price": p,
            "action": a.lower(),
            "ticker": ticker
        }
        for t, a, p in trades
    ]

    for i, seg in enumerate(sessions[:4]):
        seg_start = seg.index.min()
        seg_end = seg.index.max()

        # Сделки только в пределах текущего графика
        segment_trades = [
            {
                "timestamp": trade["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "price": trade["price"],
                "action": trade["action"],
                "ticker": trade["ticker"],
                "model": model,
            }
            for trade in parsed_trades
            if seg_start <= trade["timestamp"] <= seg_end
        ]

        await plot_trades(
            user_id=f"{user_id}_part{i + 1}",
            security_board=security_board,
            ticker=ticker,
            time_frame=time_frame,
            trade_data_container=segment_trades,
            start_time=seg_start.strftime("%d.%m.%Y %H:%M"),
            end_time=seg_end.strftime("%d.%m.%Y %H:%M"),
            bars_limit=None, # Не ограничиваем, т.к. уже отфильтровано по времени
            model=model
        )



if __name__ == '__main__':
    model_name = input("Введи Random_Forest, MACD, SMA, LSTM, LSTM_PPO, LSTM_PPO_1\n")
    print(Config.portfolio)
    ticker_name = input("Введи одну из акций: ")
    data_file = f"Data/TQBR.{ticker_name}_M5.txt"
    start_balance = 100000
    commission = 0.0005
    start_date = "15.12.2024"
    end_date = "01.04.2025"

    df = pd.read_csv(data_file, sep="\t")
    df.columns = df.columns.str.lower()
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M")
    df = df.set_index("datetime")
    df = df[(df.index >= pd.to_datetime(start_date, format="%d.%m.%Y")) &
            (df.index <= pd.to_datetime(end_date, format="%d.%m.%Y"))]

    kwargs = {}
    path = ""
    if model_name == "Random_Forest":
        kwargs["model_path"] = "Models/Random_Forest/random_forest_model_old.pkl"
    elif model_name == "LSTM":
        kwargs["model_path"] = "Models/LSTM/lstm_improved_moded_model.h5"
    elif model_name == "LSTM_PPO":
        kwargs["model_path"] = "Models/LSTM/LSTM_PPO/ppo_lstm_gazp_model.zip"
    elif model_name == "LSTM_PPO_1":
        path = "ppo_lstm_gazp_1_model"
        kwargs["model_path"] = "Models/LSTM/LSTM_PPO/ppo_lstm_checkpoints/best_of_thebest.zip"
            #"Models/LSTM/LSTM_PPO/ppo_lstm_checkpoints/ppo_lstm_checkpoint_870000_steps.zip"
            # f"Models/LSTM/LSTM_PPO/{path}.zip"
    elif model_name == "Hybrid":
        path = "best_of_the_best"
        kwargs["model_path"] = f"Models/Hybrid/{path}.zip"
            #"Models/Hybrid/ppo_lstm_hybrid_checkpoints/ppo_lstm_hybrid_checkpoint_100000_steps.zip"
            # f"Models/Hybrid/{path}.zip"

    strategy = get_strategy(model_name, **kwargs)

    os.makedirs("Simulate_data", exist_ok=True)

    balance_df, trades = backtest(
        df=df,
        strategy=strategy,
        start_balance=start_balance,
        commission=commission,
        ticker=ticker_name,
        model_name=model_name,
        start_date_str=start_date,
        end_date_str=end_date,
        log_path=f"Simulate_data/{model_name}_stats_log.txt",
        plot_file=f"Simulate_data/{model_name}/{ticker_name}{path}_backtest_balance.png"
    )

    # Генерация графиков с точками входа и выхода
    asyncio.run(generate_trade_graphs(
        trades=trades,
        df=df,
        user_id=f"1",
        security_board="TQBR",
        ticker=ticker_name,
        model=model_name
    ))
