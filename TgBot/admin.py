import logging
from typing import Optional, Tuple, Dict
from aiogram import Router, Bot, F
from aiogram.types import Message
from aiogram.enums.chat_type import ChatType
from aiogram.enums.parse_mode import ParseMode
from aiogram.client.default import DefaultBotProperties
from my_config.Cfg import Config
from channel_registry import (
    save_channel_info,
    get_channel_info,
    list_all_channels,
    delete_channel_info,
)
from strategy_manager import start_strategy

router = Router()
logger = logging.getLogger(__name__)

# Один экземпляр Bot для всех вызовов API
bot = Bot(
    token=Config.Telegram_Token,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)


@router.message(F.text.startswith("/register_channel"))
async def register_channel(message: Message):
    """
    Универсальная команда для сохранения в реестр.
    - Внутри темы форума: /register_channel MODEL TICKER
      (chat_id определяется автоматически, thread_id = message.message_thread_id)
    - Из личного чата или обычного чата: /register_channel MODEL TICKER CHAT_ID:THREAD_ID
    """
    parts = message.text.split(maxsplit=3)
    chat = message.chat

    # 1) Вызываем внутри форума
    if getattr(chat, "is_forum", False) and message.message_thread_id:
        if len(parts) != 3:
            return await message.reply(
                "❌ Внутри темы: используйте `/register_channel MODEL TICKER`",
                parse_mode=None
            )
        _, model, ticker = parts
        chat_id = chat.id
        thread_id = message.message_thread_id

    # 2) Вызываем из ЛС / другого чата
    else:
        if len(parts) != 4:
            return await message.reply(
                "❌ Вне темы: используйте `/register_channel MODEL TICKER CHAT_ID:THREAD_ID`",
                parse_mode=None
            )
        _, model, ticker, chat_thread = parts
        try:
            chat_id_str, thread_id_str = chat_thread.split(":", 1)
            chat_id = int(chat_id_str)
            thread_id = int(thread_id_str)
        except ValueError:
            return await message.reply(
                "❌ Неверный формат `CHAT_ID:THREAD_ID`",
                parse_mode=None
            )

    model = model.upper()
    ticker = ticker.upper()
    save_channel_info(model, ticker, chat_id, thread_id)
    logger.info(f"[Admin] Зарегистрирован канал: {model}_{ticker} → {chat_id}:{thread_id}")

    await message.reply(
        f"✅ Зарегистрирован канал для <b>{model} {ticker}</b>:\n"
        f"• chat_id = <code>{chat_id}</code>\n"
        f"• thread_id = <code>{thread_id}</code>",
        parse_mode=ParseMode.HTML
    )


@router.message(F.text.startswith("/remove_channel"))
async def remove_channel_cmd(message: Message):
    """
    Удаляет сохранённый чат/тему из реестра и пытается удалить саму тему в Telegram.
    Формат: /remove_channel MODEL TICKER
    """
    if message.from_user.id != Config.Admin_User_Id:
        return await message.answer("❌ У вас нет прав на эту команду.")

    parts = message.text.split(maxsplit=2)
    if len(parts) != 3:
        return await message.answer("❌ Формат: /remove_channel MODEL TICKER")

    _, model, ticker = parts
    model = model.upper()
    ticker = ticker.upper()

    info = get_channel_info(model, ticker)
    if not info:
        return await message.answer(f"❌ Для {model} {ticker} ничего не найдено в реестре.")

    chat_id, thread_id = info

    # 1) Удаляем тему в Telegram (если возможно)
    try:
        await bot.delete_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
        logger.info(f"[Admin] Удалена тема в Telegram: {model}_{ticker} → {chat_id}:{thread_id}")
    except Exception as e:
        logger.warning(f"[Admin] Не удалось удалить тему в Telegram: {e}")

    # 2) Удаляем из реестра
    deleted = delete_channel_info(model, ticker)
    if deleted:
        await message.answer(f"✅ Запись для {model} {ticker} удалена из реестра.")
    else:
        await message.answer(f"⚠️ Не удалось найти запись для {model} {ticker} при удалении.")


@router.message(F.text.startswith("/run_strategy"))
async def run_strategy_cmd(message: Message):
    """
    Админская команда для запуска live-стратегии.
    Формат: /run_strategy MODEL TICKER BALANCE
    """
    if message.from_user.id != Config.Admin_User_Id:
        return await message.answer("❌ У вас нет прав на эту команду.")

    parts = message.text.split(maxsplit=3)
    if len(parts) != 4:
        return await message.answer("❌ Формат: /run_strategy MODEL TICKER BALANCE")

    _, model, ticker, balance_str = parts
    model = model
    ticker = ticker.upper()
    try:
        balance = float(balance_str)
    except ValueError:
        return await message.answer("❌ Неверное значение BALANCE.")

    # Получаем сохранённую тему
    info = get_channel_info(model, ticker)
    if not info:
        return await message.answer(
            "❌ Канал/тема не зарегистрированы. Выполните `/register_channel`."
        )
    chat_id, thread_id = info

    # Запускаем стратегию
    await start_strategy(
        user_id=str(message.from_user.id),
        model_name=model,
        ticker=ticker,
        balance=balance,
        chat_id=chat_id,
        thread_id=thread_id
    )

    await message.answer(
        f"🚀 Стратегия <b>{model} {ticker}</b> запущена.\n"
        f"• chat_id = <code>{chat_id}</code>\n"
        f"• thread_id = <code>{thread_id}</code>",
        parse_mode=ParseMode.HTML
    )


@router.message(F.text == "/list_channels")
async def list_channels_handler(message: Message):
    """
    Показывает все зарегистрированные стратегии и их форумы.
    """
    if message.from_user.id != Config.Admin_User_Id:
        return await message.answer("❌ У вас нет прав на эту команду.")

    registry = list_all_channels()
    if not isinstance(registry, dict) or not registry:
        return await message.answer("📭 Реестр пуст.")

    text = "📋 <b>Зарегистрированные стратегии и каналы:</b>\n\n"
    for key, val in registry.items():
        model, ticker = key.split("_", 1)
        text += (
            f"• <b>{model} {ticker}</b>\n"
            f"  – chat_id: <code>{val['chat_id']}</code>\n"
            f"  – thread_id: <code>{val['thread_id']}</code>\n\n"
        )

    await message.answer(text)


@router.message(F.text == "/chatinfo")
async def chat_info_handler(message: Message):
    """
    Показывает chat_id и thread_id текущего чата и темы.
    """
    chat_id = message.chat.id
    thread_id = message.message_thread_id

    await message.answer(
        f"<b>chat_id</b>: <code>{chat_id}</code>\n"
        f"<b>thread_id</b>: <code>{thread_id}</code>",
        parse_mode=ParseMode.HTML
    )