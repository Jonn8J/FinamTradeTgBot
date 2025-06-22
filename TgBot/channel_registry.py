import json
import os
from typing import Dict, Optional, Tuple

CHANNELS_FILE = "Data/channel_registry.json"


def _load_registry() -> Dict[str, dict]:
    """
    Загружает реестр из файла. Если файл не существует или содержит не-словарь, возвращает пустой словарь.
    """
    if not os.path.exists(CHANNELS_FILE):
        return {}
    with open(CHANNELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Гарантируем, что реестр — именно dict
    if not isinstance(data, dict):
        return {}
    return data


def _save_registry(registry: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(CHANNELS_FILE), exist_ok=True)
    with open(CHANNELS_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def save_channel_info(model: str, ticker: str, chat_id: int, thread_id: int) -> None:
    """
    Сохраняет информацию о канале и теме для стратегии.
    """
    key = f"{model.upper()}_{ticker.upper()}"
    registry = _load_registry()
    registry[key] = {"chat_id": chat_id, "thread_id": thread_id}
    _save_registry(registry)


def get_channel_info(model: str, ticker: str) -> Optional[Tuple[int, int]]:
    """
    Получает (chat_id, thread_id) по модели и тикеру, или None.
    """
    key = f"{model.upper()}_{ticker.upper()}"
    registry = _load_registry()
    if key in registry and isinstance(registry[key], dict):
        info = registry[key]
        return info.get("chat_id"), info.get("thread_id")
    return None


def delete_channel_info(model: str, ticker: str) -> bool:
    """
    Удаляет запись из реестра. Возвращает True, если была запись и она удалена.
    """
    key = f"{model.upper()}_{ticker.upper()}"
    registry = _load_registry()
    if key in registry:
        del registry[key]
        _save_registry(registry)
        return True
    return False


def list_all_channels() -> Dict[str, dict]:
    """
    Возвращает весь реестр в виде словаря.
    """
    return _load_registry()
