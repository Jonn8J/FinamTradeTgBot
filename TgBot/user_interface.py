import logging
import os
import pandas as pd
from datetime import datetime
from aiogram import Router, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
)
from aiogram.enums.parse_mode import ParseMode
from Models.strategy_registry import get_strategy
from Simulation import backtest, generate_trade_graphs
from request_manager import RequestManager
from model_config import get_model_keyboard, get_stock_keyboard
from aiogram.types import FSInputFile
from my_config.Cfg import Config

router = Router()
logger = logging.getLogger(__name__)

user_data = {}
stock_codes = {
    'Сбербанк': 'SBER', 'ВТБ': 'VTBR', 'Газпром': 'GAZP', 'Сургутнефтегаз': 'SNGS',
    'Татнефть': 'TATN', 'Лукойл': 'LKOH', 'Норникель': 'GMKN', 'Роснефть': 'ROSN', 'Новатэк': 'NVTK'
}


@router.message(F.text == "/start")
async def start_handler(message: Message):
    user_id = str(message.chat.id)
    user_data[user_id] = {"user_id": user_id}

    welcome_text = (
        "👋 Добро пожаловать в симулятор торговли FinamBot!\n\n"
        "Вы можете:\n"
        "📊 Запустить симуляцию торговли\n"
        "📢 Подписаться на сигналы ИИ в Telegram-каналах\n\n"
        "Выберите действие:"
    )

    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Запустить симуляцию", callback_data="start_simulation")],
        [InlineKeyboardButton(text="📰 Перейти в новостной канал", url="https://t.me/your_channel_here")]
    ])

    await message.answer(welcome_text, reply_markup=markup)


@router.callback_query(F.data == "start_simulation")
async def start_simulation(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    user_data[user_id]["step"] = "awaiting_amount"
    await callback.message.edit_text("Введите сумму для симуляции (например, 100000):")


@router.message(F.text.regexp(r"^\d+([.,]\d+)?$"))
async def process_amount(message: Message):
    user_id = str(message.chat.id)
    if user_data.get(user_id, {}).get("step") != "awaiting_amount":
        return

    try:
        amount = float(message.text.replace(",", "."))
        if amount <= 0:
            raise ValueError
        user_data[user_id]["trading_amount"] = amount
        user_data[user_id]["step"] = "awaiting_ticker"

        await message.answer("📈 Выберите акцию для торговли:", reply_markup=get_stock_keyboard())
    except ValueError:
        await message.answer("⚠️ Введите корректную сумму (например, 100000).")


@router.callback_query(F.data.startswith("ticker:"))
async def choose_ticker(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    ticker_name = callback.data.split("ticker:")[1]
    user_data[user_id]["ticker"] = stock_codes[ticker_name]
    user_data[user_id]["step"] = "awaiting_model"

    await callback.message.edit_text(f"📈 Выбранная акция: {ticker_name}")
    await callback.message.answer("🧠 Выберите торговую стратегию:", reply_markup=get_model_keyboard())


@router.callback_query(F.data.startswith("model:"))
async def choose_model(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    model_name = callback.data.split("model:")[1]
    user_data[user_id]["model_name"] = model_name
    user_data[user_id]["step"] = "awaiting_start_date"

    await callback.message.edit_text(f"🧠 Выбранная стратегия: {model_name}")
    await callback.message.answer("📅 Введите дату начала торговли (например, 01.12.2024):")


@router.message(F.text.regexp(r"^\d{2}\.\d{2}\.\d{4}$"))
async def set_dates(message: Message):
    user_id = str(message.chat.id)
    step = user_data.get(user_id, {}).get("step")

    try:
        date = datetime.strptime(message.text, "%d.%m.%Y")

        if step == "awaiting_start_date":
            user_data[user_id]["start_date"] = message.text
            user_data[user_id]["step"] = "awaiting_end_date"
            await message.answer("📅 Теперь введите дату окончания торговли (например, 05.04.2025):")

        elif step == "awaiting_end_date":
            user_data[user_id]["end_date"] = message.text
            user_data[user_id]["step"] = "ready"

            await confirm_configuration(message)

    except ValueError:
        await message.answer("⚠️ Неверный формат даты. Введите в формате: 01.12.2024")


async def confirm_configuration(message: Message):
    user_id = str(message.chat.id)
    data = user_data[user_id]

    summary = (
        f"🎯 Конфигурация:\n"
        f"Акция: {data['ticker']}\n"
        f"Модель: {data['model_name']}\n"
        f"Сумма: {data['trading_amount']:.2f} руб.\n"
        f"Период: {data['start_date']} — {data['end_date']}"
    )

    markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🚀 Запустить симуляцию", callback_data="simulate_now")]
    ])
    await message.answer(summary, reply_markup=markup)


@router.callback_query(F.data == "simulate_now")
async def simulate_from_callback(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    data = user_data.get(user_id)

    if not data:
        await callback.message.answer("⚠️ Сначала заполните все параметры.")
        return

    ticker = data["ticker"]
    model_name = data["model_name"]
    balance = data["trading_amount"]
    start_date = data["start_date"]
    end_date = data["end_date"]

    await callback.message.edit_text("🔄 Получаем данные...")

    # Загружаем из файла
    df = await RequestManager.get_data("TQBR", ticker, "M5")

    if df is None or df.empty:
        await callback.message.answer("❌ Не удалось загрузить исторические данные.")
        return

    df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M")
    df.set_index("datetime", inplace=True)
    df = df[(df.index >= pd.to_datetime(start_date, format="%d.%m.%Y")) &
            (df.index <= pd.to_datetime(end_date, format="%d.%m.%Y"))]

    if df.empty:
        await callback.message.answer("⚠️ В указанный период нет доступных данных.")
        return

    kwargs = {}
    if model_name == "Random_Forest":
        kwargs["model_path"] = "../Models/Random_Forest/random_forest_model_old.pkl"
    elif model_name == "LSTM":
        kwargs["model_path"] = "../Models/LSTM/lstm_improved_moded_model.h5"
    elif model_name == "LSTM_PPO":
        kwargs["model_path"] = "../Models/LSTM/LSTM_PPO/ppo_lstm_gazp_model.zip"
    elif model_name == "LSTM_PPO_1":
        path = "ppo_lstm_gazp_1_model"
        kwargs["model_path"] = "../Models/LSTM/LSTM_PPO/ppo_lstm_checkpoints/best_of_thebest"
            #"Models/LSTM/LSTM_PPO/ppo_lstm_checkpoints/ppo_lstm_checkpoint_870000_steps.zip"
            # f"Models/LSTM/LSTM_PPO/{path}.zip"
    elif model_name == "Hybrid":
        path = "best_of_the_best"
        kwargs["model_path"] = f"../Models/Hybrid/{path}"
            #"Models/Hybrid/ppo_lstm_hybrid_checkpoints/ppo_lstm_hybrid_checkpoint_100000_steps.zip"
            # f"Models/Hybrid/{path}.zip"
    strategy = get_strategy(model_name, **kwargs)

    plot_file = f"Simulate_data/{model_name}/{ticker}_backtest_balance.png"
    log_path = f"Simulate_data/{model_name}_stats_log.txt"
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    # Запуск симуляции
    balance_df, trades = backtest(
        df=df,
        strategy=strategy,
        start_balance=balance,
        commission=0.0005,
        ticker=ticker,
        model_name=model_name,
        start_date_str=start_date,
        end_date_str=end_date,
        log_path=log_path,
        plot_file=plot_file
    )

    # Отправка графика баланса
    if os.path.exists(plot_file):
        await callback.message.answer_photo(FSInputFile(plot_file), caption="📊 График баланса")
        os.remove(plot_file)

    # Графики сделок
    await generate_trade_graphs(
        trades=trades,
        df=df,
        user_id=user_id,
        security_board="TQBR",
        ticker=ticker,
        model=model_name
    )

    # Графики сделок
    graph_dir = f"Graph/{model_name}/{ticker}"
    graph_files = sorted([
        file for file in os.listdir(graph_dir)
        if file.startswith(f"graph_{user_id}_part") and file.endswith(".png")
    ])

    if graph_files:
        await callback.message.answer("📌 *Сделки по стратегии:*", parse_mode=ParseMode.MARKDOWN)

        for file in graph_files:
            file_path = os.path.join(graph_dir, file)
            await callback.message.answer_photo(FSInputFile(file_path))
            os.remove(file_path)

    # 📌 Общие итоги
    final_balance = balance_df["balance"].iloc[-1]
    returns = balance_df["balance"].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * (252 ** 0.5) if not returns.empty else 0
    max_dd = ((balance_df["balance"].cummax() - balance_df["balance"]) / balance_df["balance"].cummax()).max()

    summary = (
        "📋 *Общие итоги симуляции:*\n"
        f"• 💰 Финальный баланс: `{final_balance:,.2f}` руб.\n"
        f"• 📉 Макс. просадка: `{max_dd:.2%}`\n"
        f"• 📈 Коэф. Шарпа: `{sharpe:.4f}`\n"
        f"• 🔄 Сделок: `{len(trades)}`"
    )

    await callback.message.answer(summary, parse_mode=ParseMode.MARKDOWN)


# Текст для обычных пользователей
USER_HELP_TEXT = (
    "🤖 Пользовательские команды FinamBot:\n"
    "/start — главное меню\n"
    "/simulate — запустить симуляцию (после настройки)\n"
    "/help — показать эту справку\n\n"
)

ADMIN_HELP_TEXT = (
    USER_HELP_TEXT +
    "📋 Админ‑команды:\n"
    "/register_channel MODEL TICKER THREAD_ID — зарегистрировать форумную тему\n"
    "/add_channel MODEL TICKER CHAT_ID:THREAD_ID — добавить вручную\n"
    "/remove_channel MODEL TICKER — удалить канал\n"
    "/list_channels — список зарегистрированных каналов\n"
    "/run_strategy MODEL TICKER BALANCE — запустить live‑стратегию\n"
    "/list_topics <chat_id> — показать темы форума"
)

# Кнопка «Помощь» в обычной клавиатуре
HELP_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Помощь")]],
    resize_keyboard=True,
    one_time_keyboard=True
)

@router.message(F.text == "/help")
@router.message(F.text == "Помощь")
async def cmd_help(message: Message):
    is_admin = (message.from_user.id == Config.Admin_User_Id)
    text = ADMIN_HELP_TEXT if is_admin else USER_HELP_TEXT
    # убираем кастомную клавиатуру, показываем Markdown‐справку
    await message.answer(
        f"```\n{text}\n```",
        parse_mode=ParseMode.MARKDOWN
    )


@router.message()  # этот хэндлер в конце, после всех остальных
async def fallback(message: Message):
    # если прилетел /start или другой известный — пропустим
    if message.text in ("/start", "/simulate", "/help", "Помощь"):
        return
    # иначе — предлагаем помощь
    await message.answer(
        "Я не понял запрос. Нажмите кнопку «Помощь», чтобы получить список команд.",
        reply_markup=HELP_KEYBOARD
    )

def save_user_data(user_id):
    os.makedirs("Data", exist_ok=True)
    path = "Data/users.txt"
    df_new = pd.DataFrame([user_data[user_id]])
    df_new.set_index("user_id", inplace=True)

    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t", index_col="user_id")
        df.update(df_new)
        df = pd.concat([df, df_new[~df_new.index.isin(df.index)]])
    else:
        df = df_new

    df.to_csv(path, sep="\t")
