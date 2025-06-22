import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

DATA_DIR = r"D:\PycharmProjects\FinamBot_Upd\Data"
#portfolio = ["SBER", 'VTBR', 'GAZP', 'TATN', 'LKOH', 'GMKN', 'ROSN', 'NVTK', 'SNGS']
portfolio = ['GAZP']
security_board = "TQBR"

SEQ_LEN = 30
FORECAST_HORIZON = 3
THRESHOLD = 0.002  # для генерации сигналов
MODEL_PATH = "lstm_v2.h5"

# === Загрузка одного тикера ===
def load_ticker(ticker):
    file_name = f"{security_board}.{ticker}_M5.txt"
    file_path = os.path.join(DATA_DIR, file_name)

    df = pd.read_csv(file_path, sep="\t")
    df.columns = df.columns.str.lower()
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


# === Добавление тех. индикаторов ===
def add_features(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
    df['macd'] = MACD(df['close']).macd_diff()
    df['bb_width'] = BollingerBands(df['close']).bollinger_wband()
    df['roc'] = ROCIndicator(df['close'], window=10).roc()
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['sma_10'] = SMAIndicator(df['close'], window=10).sma_indicator()
    df['sma_30'] = SMAIndicator(df['close'], window=30).sma_indicator()
    df = df.dropna()
    return df


def prepare_data(df, seq_len=SEQ_LEN, forecast_horizon=FORECAST_HORIZON):
    df = add_features(df.copy())  # ⬅ важно: создаём копию, чтобы избежать предупреждений
    df['future_return'] = df['close'].shift(-forecast_horizon) / df['close'] - 1

    # Создаём целевую переменную target
    df['target'] = 1  # hold
    df.loc[df['future_return'] > THRESHOLD, 'target'] = 2  # buy
    df.loc[df['future_return'] < -THRESHOLD, 'target'] = 0  # sell

    df.dropna(inplace=True)

    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'rsi', 'ema_12', 'macd', 'bb_width', 'roc', 'obv', 'sma_10', 'sma_30']

    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])

    # Генерация обучающих окон
    X, y = [], []
    for i in range(len(df) - seq_len - forecast_horizon):
        X.append(df[feature_cols].iloc[i:i + seq_len].values)
        y.append(df['target'].iloc[i + seq_len])

    return np.array(X), np.array(y)

# === Построение LSTM модели ===
def build_lstm_classifier(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 класса
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )
    return model


# === Загружаем и объединяем данные ===
X_all, y_all = [], []
for ticker in portfolio:
    try:
        df = load_ticker(ticker)
        X, y = prepare_data(df)
        X_all.append(X)
        y_all.append(y)
        print(f"[+] Loaded {ticker}: {len(y)} samples")
    except Exception as e:
        print(f"[!] Error loading {ticker}: {e}")

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
print(f"✅ Total samples: {X_all.shape[0]}")

# === Учёт дисбаланса классов ===
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_all),
    y=y_all
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# === Разбивка на тренировку и тест ===
split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# === Создаём и обучаем модель ===
model = build_lstm_classifier(X_all.shape[1:])

# Сохраняем лучшую модель по валидации
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_sparse_categorical_accuracy',
                             save_best_only=True, mode='max', verbose=1)

model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[checkpoint]
)

# === Загружаем лучшую модель и проверяем предсказания ===
model.load_weights(MODEL_PATH)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=["sell", "hold", "buy"]))

# === Вывод первых сигналов ===
for i in range(10):
    action = ["sell", "hold", "buy"][y_pred[i]]
    print(f"Predicted: {action}, True: {['sell', 'hold', 'buy'][y_test[i]]}")