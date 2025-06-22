import asyncio
import logging

from Live_strategy import LiveStrategy
from channel_registry import get_channel_info

# Централизованное хранилище всех активных стратегий
active_strategies = {}

logger = logging.getLogger("StrategyManager")


async def start_strategy(
    user_id: str,
    model_name: str,
    ticker: str,
    timeframe: str = "M5",
    balance: float = 100000,
    chat_id: int | None = None,
    thread_id: int | None = None
):
    """
    Запускает LiveStrategy в фоне.
    Если chat_id/thread_id не переданы, попробует взять их из registry.
    """
    strategy_id = f"{user_id}_{model_name}_{ticker}"

    if strategy_id in active_strategies:
        logger.warning(f"Стратегия уже запущена: {strategy_id}")
        return

    # Если не передали канал/тему — попытаемся вытянуть из registry
    if chat_id is None or thread_id is None:
        info = get_channel_info(model_name, ticker)
        if info:
            chat_id, thread_id = info
        else:
            logger.warning(f"Не найдена регистрация канала для {model_name}_{ticker}")

    # Создаём экземпляр стратегии с параметрами
    strategy = LiveStrategy(
        user_id=user_id,
        model_name=model_name,
        ticker=ticker,
        timeframe=timeframe,
        balance=balance,
        chat_id=chat_id,
        thread_id=thread_id
    )

    # Запускаем в фоне loop.check_for_signal()
    task = asyncio.create_task(strategy.start())
    active_strategies[strategy_id] = {
        "strategy": strategy,
        "task": task
    }

    logger.info(f"Стратегия {strategy_id} запущена.")


async def stop_strategy(user_id: str, model_name: str, ticker: str):
    """
    Останавливает ранее запущенную стратегию.
    """
    strategy_id = f"{user_id}_{model_name}_{ticker}"

    if strategy_id not in active_strategies:
        logger.warning(f"Нет активной стратегии: {strategy_id}")
        return

    entry = active_strategies[strategy_id]
    await entry["strategy"].stop()
    entry["task"].cancel()
    del active_strategies[strategy_id]

    logger.info(f"Стратегия {strategy_id} остановлена.")


def get_active_strategy_ids() -> list[str]:
    """
    Возвращает список идентификаторов запущенных стратегий.
    """
    return list(active_strategies.keys())
