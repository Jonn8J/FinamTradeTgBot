import os


class Config:

    ClientIds = ('',)  # Торговые счета
    AccessToken = ''  # Торговый токен

    Telegram_Token = '' # Токен тг бота
    Admin_User_Name = '' # Ваше имя в тг
    Admin_User_Id = 1 # Ваш тг id

    FilePath = "Data/"
    # FilePath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data')) + os.sep
    portfolio = ["SBER", 'VTBR',  'GAZP', 'TATN',  'LKOH',  'GMKN',  'ROSN',  'NVTK', 'SNGS']  # тикеры по которым торгуем и скачиваем исторические данные
    security_board = "TQBR"  # класс тикеров

    # доступные M1, M5, M15, H1
    timeframe_0 = "M1"  # таймфрейм
    timeframe_1 = "M5"  # таймфрейм
    start = "2021-01-01"  # с какой даты загружаем исторические данные

    live_check_interval = 10

    trading_hours_start = "7:00"  # время работы биржи - начало
    trading_hours_end = "23:50"  # время работы биржи - конец
