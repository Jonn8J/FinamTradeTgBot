from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Вернуть серию сигналов ('buy', 'sell', 'hold') длиной в df"""
        pass
