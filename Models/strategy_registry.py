from .SMA.SMA_strategy import SMAStrategy
from .Random_Forest.RF_strategy import RandomForestStrategy
from .MACD.MACD_strategy import MACDStrategy
from .LSTM.lstm_strategy import LSTMStrategy
#from .LSTM.lstm_ppo_strategy import LSTMPPOStrategy
from .LSTM.lstm_ppo_strategy_1_ import LSTMPPOStrategy
#from .LSTM.lstm_ppo_strategy_env import LSTMPPOStrategy
from .Hybrid.lstm_ppo_hybrid_strategy import LSTMPPOHybridStrategy

strategies = {
    # для SMA игнорируем все переданные kwargs
    "SMA":   lambda **kwargs: SMAStrategy(),

    # для Random Forest — заберём model_path из kwargs
    "Random_Forest": lambda model_path=None, **kwargs: RandomForestStrategy(model_path=model_path),

    # аналогично для других моделей, если они требуют model_path:
    "MACD":  lambda **kwargs: MACDStrategy(),

    "LSTM": lambda model_path=None, **kwargs: LSTMStrategy(model_path=model_path),

    #"LSTM_PPO": lambda model_path=None, **kwargs: LSTMPPOStrategy(model_path=model_path),

    #"LSTM_PPO_1": lambda model_path=None, **kwargs: LSTMPPOStrategy(model_path=model_path),

    "Hybrid": lambda model_path=None, **kwargs: LSTMPPOHybridStrategy(model_path=model_path),

}

def get_strategy(name: str, **kwargs):
    try:
        return strategies[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown strategy {name}")
