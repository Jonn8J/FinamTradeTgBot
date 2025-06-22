import asyncio
import logging
from datetime import datetime, timedelta

from Bars import save_candles_to_file, load_candles_from_file
from FinamPy import FinamPy
from my_config.Cfg import Config

logger = logging.getLogger("RequestManager")


class RequestManager:
    _lock = asyncio.Lock()
    _last_updated = {}      # (board, ticker, timeframe): datetime
    _data_cache = {}        # (board, ticker, timeframe): pd.DataFrame
    _price_cache = {}       # (board, ticker): {"ask": float, "bid": float}
    _cache_ttl = timedelta(minutes=5)

    @classmethod
    async def get_data(cls, board: str, ticker: str, timeframe: str):
        key = (board, ticker, timeframe)
        await cls._cleanup_cache()

        async with cls._lock:
            now = datetime.now()
            last_update = cls._last_updated.get(key)

            if not last_update or (now - last_update > timedelta(seconds=30)):
                logger.debug(f"Обновление данных с биржи для {ticker} [{timeframe}]")

                try:
                    provider = FinamPy(Config.ClientIds, Config.AccessToken)
                    save_candles_to_file(
                        provider,
                        class_code=board,
                        security_codes=(ticker,),
                        tf=timeframe,
                        skip_last_date=False
                    )
                    provider.close_channel()
                    cls._last_updated[key] = now
                except Exception as e:
                    logger.warning(f"Не удалось получить данные с биржи для {ticker}: {e}")

            try:
                logger.debug(f"Загружаем файл из: Data/{board}.{ticker}_{timeframe}.txt")
                df = load_candles_from_file(board, ticker, timeframe)
                cls._data_cache[key] = df
            except Exception as e:
                logger.error(f"Ошибка при загрузке файла {ticker}: {e}")
                df = cls._data_cache.get(key)

            return df.copy() if df is not None else None

    @classmethod
    async def get_price(cls, fp_provider, ticker: str, board: str, side: str = 'asks') -> float:
        """
        Получить цену из стакана (ask или bid)
        """
        assert side in ['asks', 'bids'], "side должен быть 'asks' или 'bids'"
        key = (board, ticker)

        price = None

        def on_order_book(order_book):
            nonlocal price
            if getattr(order_book, side):
                price = getattr(order_book, side)[0].price
                cls._price_cache[key] = {
                    "ask": order_book.asks[0].price if order_book.asks else None,
                    "bid": order_book.bids[0].price if order_book.bids else None,
                    "timestamp": datetime.now()
                }

        fp_provider.on_order_book = on_order_book
        fp_provider.subscribe_order_book(ticker, board, 'orderbook1')

        try:
            for _ in range(20):  # ждём максимум 2 секунды
                await asyncio.sleep(0.1)
                if price is not None:
                    break
        finally:
            fp_provider.unsubscribe_order_book('orderbook1', ticker, board)

        if price is None:
            logger.warning(f"Не удалось получить цену {side} для {ticker}")
            cached = cls._price_cache.get(key)
            if cached and side in cached:
                return float(cached[side])
            return 0.0

        return round(float(price), 2)

    @classmethod
    async def _cleanup_cache(cls):
        """
        Удаляет устаревшие записи из кэша (по времени).
        """
        now = datetime.now()

        # Очистка данных
        expired_data = [key for key, ts in cls._last_updated.items() if now - ts > cls._cache_ttl]
        for key in expired_data:
            cls._last_updated.pop(key, None)
            cls._data_cache.pop(key, None)

        # Очистка цен
        expired_prices = [key for key, val in cls._price_cache.items()
                          if now - val.get("timestamp", now) > cls._cache_ttl]
        for key in expired_prices:
            cls._price_cache.pop(key, None)