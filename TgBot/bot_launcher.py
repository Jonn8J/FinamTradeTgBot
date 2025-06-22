import logging
import sys
import asyncio

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.enums.parse_mode import ParseMode

from my_config.Cfg import Config
from TgBot import user_interface, admin  # обязательно подключите ваш admin.py тоже

async def main():
    # Конфигурируем логгирование один раз в самом начале
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log", encoding="utf-8", mode="a")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("🚀 Бот запускается...")

    bot = Bot(
        token=Config.Telegram_Token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher(storage=MemoryStorage())

    # Подключаем все роутеры (в том числе admin)
    dp.include_router(admin.router)
    dp.include_router(user_interface.router)

    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.getLogger(__name__).info("🛑 Бот остановлен.")
