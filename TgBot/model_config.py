from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# Список доступных моделей
AVAILABLE_MODELS = {
    "SMA": "Простое скользящее среднее",
    "Random_Forest": "Случайный лес",
    "Hybrid": "Гибридная",
    "LSTM": "LSTM‑сеть"
    # Добавь другие модели при необходимости
}

# Доступные акции
STOCK_NAMES = {
    "Сбербанк": "SBER",
    "ВТБ": "VTBR",
    "Газпром": "GAZP",
    "Сургутнефтегаз": "SNGS",
    "Татнефть": "TATN",
    "Лукойл": "LKOH",
    "Норникель": "GMKN",
    "Роснефть": "ROSN",
    "Новатэк": "NVTK"
}


def get_model_keyboard() -> InlineKeyboardMarkup:
    keyboard = []
    for code, label in AVAILABLE_MODELS.items():
        keyboard.append([InlineKeyboardButton(text=f"{code} — {label}", callback_data=f"model:{code}")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def get_stock_keyboard() -> InlineKeyboardMarkup:
    keyboard = []
    for name in STOCK_NAMES:
        keyboard.append([InlineKeyboardButton(text=name, callback_data=f"ticker:{name}")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)
