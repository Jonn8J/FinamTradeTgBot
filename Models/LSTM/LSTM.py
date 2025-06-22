import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==== Параметры ====
FILE_PATH = "../../Data/TQBR.GAZP_M5.txt"
SEQUENCE_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 64
THRESH_BUY = 0.001
THRESH_SELL = -0.001
FUTURE_SHIFT = 5
METRICS_PATH = "metrics"
MODEL_PATH = "lstm_improved_moded_model.h5"

os.makedirs(METRICS_PATH, exist_ok=True)

# ==== Загрузка и подготовка данных ====
df = pd.read_csv(FILE_PATH, sep="\t")
df.columns = df.columns.str.lower()
df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M")
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)

features = ["open", "high", "low", "close", "volume"]
df = df[features]

df["future_return"] = df["close"].shift(-FUTURE_SHIFT) / df["close"] - 1

def label_direction(x):
    if x > THRESH_BUY:
        return 2  # BUY
    elif x < THRESH_SELL:
        return 0  # SELL
    else:
        return 1  # HOLD

df["target"] = df["future_return"].apply(label_direction)
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])
X_all, y_all = [], []

for i in range(SEQUENCE_LENGTH, len(df)):
    X_all.append(scaled_features[i-SEQUENCE_LENGTH:i])
    y_all.append(df["target"].iloc[i])

X_all = np.array(X_all)
y_all_cat = to_categorical(y_all, num_classes=3)

split_idx = int(0.8 * len(X_all))
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all_cat[:split_idx], y_all_cat[split_idx:]
y_train_labels = np.argmax(y_train, axis=1)

# ==== Взвешивание классов ====
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights))

# ==== Модель ====
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQUENCE_LENGTH, len(features))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.1))
model.add(Dense(16, activation='swish', kernel_regularizer=l2(0.001)))
model.add(Dense(3, activation='softmax'))

# Было
#model = Sequential()
#model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQUENCE_LENGTH, len(features))))
#model.add(Dropout(0.2))
#model.add(Bidirectional(LSTM(32)))
#model.add(Dropout(0.1))
#model.add(Dense(16, activation='swish', kernel_regularizer=l2(0.001)))
#model.add(Dense(3, activation='softmax'))

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ==== Обучение ====
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

model.save(MODEL_PATH)

# ==== Графики ====
# Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(f"{METRICS_PATH}/lstm_improved_moded_accuracy.png")
plt.close()

# Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f"{METRICS_PATH}/lstm_improved_moded_loss.png")
plt.close()

# Confusion Matrix
y_pred_classes = model.predict(X_test).argmax(axis=1)
y_true_classes = y_test.argmax(axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sell', 'Hold', 'Buy'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{METRICS_PATH}/lstm_improved_moded_confusion_matrix.png")
plt.close()
