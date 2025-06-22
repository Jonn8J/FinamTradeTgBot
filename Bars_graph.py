import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from mplfinance.original_flavor import candlestick_ohlc
import aiofiles
import asyncio
from io import StringIO
from my_config.Cfg import Config
import os


async def read_file_async(file_path):
    async with aiofiles.open(file_path, mode='r') as file:
        return await file.read()


async def plot_trades(user_id, security_board, ticker, time_frame, trade_data_container,
                      start_time=None, end_time=None, bars_limit=100, model="Unknown"):
    """
    Рисует график одной акции с сигналами стратегии.
    """
    file_path = f'{Config.FilePath}{security_board}.{ticker}_{time_frame}.txt'
    raw_data = await read_file_async(file_path)
    df = pd.read_csv(StringIO(raw_data), delimiter='\t')
    df.columns = df.columns.str.lower()
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M')

    # Фильтрация
    if start_time:
        df = df[df['datetime'] >= pd.to_datetime(start_time, format="%d.%m.%Y %H:%M")]
    if end_time:
        df = df[df['datetime'] <= pd.to_datetime(end_time, format="%d.%m.%Y %H:%M")]

    df = df.tail(bars_limit) if bars_limit else df
    if df.empty:
        print(f"Нет данных для {ticker} после фильтрации.")
        return

    df['num'] = mdates.date2num(df['datetime'])
    ohlc = df[['num', 'open', 'high', 'low', 'close']]

    fig, ax = plt.subplots(figsize=(16, 9))
    candlestick_ohlc(ax, ohlc.values, width=0.002, colorup='green', colordown='red', alpha=0.8)

    for trade in trade_data_container:
        if trade.get("ticker") != ticker or trade.get("model") != model:
            continue
        time = mdates.date2num(pd.to_datetime(trade['timestamp']))
        price = trade['price']
        marker = '^' if trade['action'] == 'buy' else 'v'
        color = 'limegreen' if trade['action'] == 'buy' else 'tomato'
        ax.plot(time, price, marker=marker, color=color, markersize=10, markeredgecolor='black', label=trade['action'].capitalize())

    ax.set_xlabel("Время")
    ax.set_ylabel("Цена")
    ax.set_title(f'{ticker} — {df["datetime"].min():%d.%m %H:%M} ➜ {df["datetime"].max():%d.%m %H:%M}')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=20, prune='both'))
    fig.autofmt_xdate()
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys())

    # Путь сохранения
    save_dir = f"Graph/{model}/{ticker}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/graph_{user_id}.png"

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"График сохранён: {save_path}")
    return save_path


# Пример использования
if __name__ == "__main__":
    async def run():
        user_id = "test"
        security_board = "TQBR"
        portfolio = ["GAZP",]  # один или несколько тикеров
        time_frame = "M5"
        trade_data_container = []
        await plot_trades(
            user_id=user_id,
            security_board=security_board,
            portfolio=portfolio,
            time_frame=time_frame,
            trade_data_container=trade_data_container,
            start_time="04.04.2025 7:00",
            end_time="04.04.2025 23:50",
            bars_limit=101,
        )

    asyncio.run(run())
