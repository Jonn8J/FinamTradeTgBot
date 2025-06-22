import logging
from aiogram import Bot
from channel_registry import list_all_channels

logger = logging.getLogger(__name__)

# Уведомляет пользователя в ЛС о доступных стратегиях
async def notify_user_about_channel(bot: Bot, user_id: int):
    try:
        channels = list_all_channels()
        if not channels:
            await bot.send_message(user_id, "📭 Пока нет доступных стратегий.")
            return

        msg = "👋 Добро пожаловать!\n\n📡 Доступные стратегии и каналы:\n\n"
        for key, value in channels.items():
            model, ticker = key.split("_")
            chat_id = value.get("chat_id")
            thread_id = value.get("thread_id")

            msg += (
                f"🔸 *{model} {ticker}*\n"
                f"• `chat_id`: `{chat_id}`\n"
                f"• `thread_id`: `{thread_id}`\n\n"
            )

        msg += "Вы можете следить за сигналами в указанных каналах или попробовать симуляцию торговли здесь 👇"
        await bot.send_message(user_id, msg, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"[Welcome] Ошибка при отправке приветствия пользователю {user_id}: {e}")
