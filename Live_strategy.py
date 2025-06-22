import asyncio
import logging
import os
from datetime import datetime
import pandas as pd
from FinamPy import FinamPy

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.types import FSInputFile

from Bars_graph import plot_trades
from request_manager import RequestManager
from my_config.Cfg import Config
from Models.strategy_registry import get_strategy
from TgBot.channel_registry import get_channel_info
import traceback

logger = logging.getLogger("LiveStrategy")

bot = Bot(
    token=Config.Telegram_Token,
    default=DefaultBotProperties(parse_mode="HTML"),
)


async def broadcast_news(model: str, ticker: str, text: str, photo: FSInputFile = None):
    """
    Универсальная функция для рассылки сообщений/фото в зарегистрированную тему форума.
    Модель и тикер в реестре должны быть уже сохранены через /add_channel.
    """
    logger.info("broadcast_news: модель=%s, тикер=%s", model, ticker)
    info = get_channel_info(model, ticker)
    logger.info("channel info: %s", info)

    if not info:
        logger.warning("Не найден канал для %s %s", model, ticker)
        return

    chat_id, thread_id = info
    try:
        if photo:
            await bot.send_photo(
                chat_id=chat_id,
                message_thread_id=thread_id,
                photo=photo,
                caption=text,
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                message_thread_id=thread_id,
                text=text,
            )
        logger.info("broadcast_news: отправка прошла успешно")
    except Exception as e:
        logger.error("Ошибка отправки broadcast_news: %s", e)

def get_model(model_name):
    path = ''
    if model_name == "Random_Forest":
        path = "../Models/Random_Forest/random_forest_model_old.pkl"
    elif model_name == "LSTM":
        path = "../Models/LSTM/lstm_improved_moded_model.h5"
    elif model_name == "Hybrid":
        path = "../Models/Hybrid/best_of_the_best"

    return path

class LiveStrategy:
    def __init__(
        self,
        user_id: str,
        model_name: str,
        ticker: str,
        timeframe: str = 'M5',
        balance: float = 100000,
        chat_id: int = None,
        thread_id: int = None,
    ):
        self.user_id = user_id
        self.model_name = model_name
        self.ticker = ticker
        self.timeframe = timeframe
        self.balance = balance
        self.strategy = get_strategy(
            model_name,
            model_path= get_model(model_name)
        )
        logger.info("Стратегия успешно загружена: %s", model_name)
        self.security_board = Config.security_board
        self.in_position = False
        self.entry_price = 0.0
        self.quantity = 0
        self.trades = []
        self.stop_event = asyncio.Event()
        self.stop_loss = 0.0

        # Если передали вручную — ок, иначе подтянем из channel_registry
        if chat_id and thread_id:
            self.chat_id = chat_id
            self.thread_id = thread_id
        else:
            info = get_channel_info(model_name, ticker)
            self.chat_id, self.thread_id = info if info else (None, None)

    async def get_realtime_prices(self, retries: int = 5, delay: float = 1.0):
        provider = FinamPy(Config.ClientIds, Config.AccessToken)

        for attempt in range(retries):
            ask = await RequestManager.get_price(provider, self.ticker, self.security_board, 'asks')
            bid = await RequestManager.get_price(provider, self.ticker, self.security_board, 'bids')

            if ask and ask > 0 and bid and bid > 0:
                provider.close_channel()
                return ask, bid

            logger.warning(f"Невалидные ask/bid цены (попытка {attempt + 1}/{retries}): ask={ask}, bid={bid}")
            await asyncio.sleep(delay)

        provider.close_channel()
        logger.error("Не удалось получить валидные ask/bid после нескольких попыток.")
        return None, None

    async def check_for_signal(self):
        logger.info("Цикл check_for_signal() начал работу")
        df = await RequestManager.get_data(self.security_board, self.ticker, self.timeframe)
        df.columns = df.columns.str.lower()
        df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M")
        df = df.set_index("datetime")

        if df is None or df.empty:
            logger.warning("Нет данных по %s", self.ticker)
            return
        else:
            logger.info("Получено %d строк", len(df))

        df = df.tail(120)
        signals = self.strategy.generate_signals(df)
        signal = signals.iloc[-1] if not signals.empty else 'hold'

        logger.info(f"[DEBUG] Сигнал от модели: {signal}")
        logger.info("%s %s → %s", self.model_name, self.ticker, signal)

        now = datetime.now()

        # Вход
        if not self.in_position and signal == "buy":
            self.ask_price, _ = await self.get_realtime_prices()
            if not self.ask_price:
                logger.warning("Пропуск покупки: нет актуальной цены.")
                return
            self.quantity = int(self.balance // self.ask_price)
            self.balance -= self.quantity * self.ask_price
            self.stop_loss = self.ask_price * 0.99
            self.in_position = True
            self.trades.append({
                'action': 'buy', 'timestamp': now, 'price': self.ask_price,
                'quantity': self.quantity, 'ticker': self.ticker, 'model': self.model_name
            })
            logger.info("BUY %s шт по %.2f", self.quantity, self.ask_price)
            await self.send_trade_plot()

        # Выход
        elif self.in_position:
            _, self.bid_price = await self.get_realtime_prices()
            if not self.bid_price:
                logger.warning("Пропуск продажи: нет актуальной цены.")
                return
            if signal == "sell" or self.bid_price <= self.stop_loss:
                self.balance += self.quantity * self.bid_price
                self.trades.append({
                    'action': 'sell', 'timestamp': now, 'price': self.bid_price,
                    'quantity': self.quantity, 'ticker': self.ticker, 'model': self.model_name
                })
                logger.info("SELL %s шт по %.2f", self.quantity, self.bid_price)
                await self.send_trade_plot()
                self.in_position = False
                self.quantity = 0

    async def send_trade_plot(self):
        # 1) отрисуем локально
        await plot_trades(
            user_id=self.user_id,
            security_board=self.security_board,
            ticker=self.ticker,
            time_frame=self.timeframe,
            trade_data_container=self.trades,
            model=self.model_name
        )

        file_path = f"Graph/{self.model_name}/{self.ticker}/graph_{self.user_id}.png"
        if not os.path.exists(file_path):
            return

        last = self.trades[-1]
        action = last['action'].upper()
        price = last['price']
        qty = last['quantity']
        ts = last['timestamp'].strftime('%d.%m.%Y %H:%M')

        caption = (
            f"📢 <b>{self.model_name}</b> | <b>{self.ticker}</b>\n"
            f"🕒 {ts}\n"
            f"{'🚀 BUY' if action=='BUY' else '🔻 SELL'} {qty}шт по {price:.2f}₽"
        )

        photo = FSInputFile(file_path)
        # 2) рассылаем в зарегистрированную тему
        await broadcast_news(self.model_name, self.ticker, caption, photo)

        # 3) чистим локальный файл
        os.remove(file_path)
        logger.info("Файл удалён: %s", file_path)

    async def start(self):
        # при старте тоже можно оповестить
        logger.info("Метод start() запущен")
        logger.info("stop_event.is_set(): %s", self.stop_event.is_set())
        logger.info("Получен chat_id: %s, thread_id: %s", self.chat_id, self.thread_id)
        await broadcast_news(
            self.model_name,
            self.ticker,
            f"🚀 Старт стратегии <b>{self.model_name}</b> по <b>{self.ticker}</b>"
        )
        logger.info("Старт %s %s", self.model_name, self.ticker)

        while not self.stop_event.is_set():
            logger.info("check_for_signal() ЦИКЛ ЗАПУЩЕН")
            try:
                logger.info("Вызов check_for_signal()")
                await self.check_for_signal()
            except Exception as e:
                import traceback
                logger.error("Ошибка в стратегии: %s\n%s", e, traceback.format_exc())
            await asyncio.sleep(30)

    async def stop(self):
        self.stop_event.set()
        await broadcast_news(
            self.model_name,
            self.ticker,
            f"Остановка стратегии <b>{self.model_name}</b> по <b>{self.ticker}</b>"
        )
        logger.info("Стоп %s %s", self.model_name, self.ticker)
