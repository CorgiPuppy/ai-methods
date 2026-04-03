import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from PIL import Image, ImageOps # Для работы с твоими фото

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential
from keras.layers import (Dense, Dropout, Flatten,
                          Conv2D, MaxPool2D, BatchNormalization, Input)
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Отключаем системные логи TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'train.csv')
MY_NUMBERS_PATH = os.path.join(BASE_DIR, '..', 'data', 'my_numbers')
ASSETS_DIR = os.path.join(BASE_DIR, '..', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

def save_txt(filename, content):
    with open(os.path.join(ASSETS_DIR, filename), 'w', encoding='utf-8') as f:
        f.write(content)

def log_stage(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def run_lab():
    log_stage("Начало выполнения Лабораторной работы №2")

    # --- 1. Загрузка данных ---
    log_stage("Загрузка данных из CSV...")
    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.drop(columns=["Unnamed: 0"]) if "Unnamed: 0" in df_raw.columns else df_raw
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    save_txt('data_info.txt', buffer.getvalue())

    X = df.drop('label', axis=1).values / 255.0
    y = df['label'].values

    log_stage("Генерация средних изображений (EDA)...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for digit, ax in enumerate(axes.flatten()):
        mean_img = X[y == digit].mean(axis=0).reshape(28, 28)
        ax.imshow(mean_img, cmap='hot')
        ax.set_title(f'Digit {digit}')
        ax.axis('off')
    plt.savefig(os.path.join(ASSETS_DIR, 'mean_images.png'))
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn  = X_test.reshape(-1, 28, 28, 1)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test, 10)

    # --- 2. Эксперимент с эпохами ---
    log_stage("Запуск эксперимента с эпохами...")
    epochs_list = [3, 5, 10]
    mlp_acc, cnn_acc = [], []
    for ep in epochs_list:
        m1 = Sequential([Input(shape=(784,)), Dense(8, activation='relu'), Dense(10, activation='softmax')])
        m1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m1.fit(X_train, y_train_cat, epochs=ep, batch_size=512, verbose=0)
        mlp_acc.append(m1.evaluate(X_test, y_test_cat, verbose=0)[1])
        
        m2 = Sequential([Input(shape=(28, 28, 1)), Conv2D(2, (3, 3), activation='relu'), MaxPool2D((2, 2)), Flatten(), Dense(10, activation='softmax')])
        m2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m2.fit(X_train_cnn, y_train_cat, epochs=ep, batch_size=512, verbose=0)
        cnn_acc.append(m2.evaluate(X_test_cnn, y_test_cat, verbose=0)[1])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, mlp_acc, label='MLP', marker='o')
    plt.plot(epochs_list, cnn_acc, label='CNN', marker='s')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(ASSETS_DIR, 'epochs_impact.png'))
    plt.close()

    # --- 3. Финальное обучение ---
    log_stage("Финальное обучение моделей...")
    mlp_model = Sequential([Input(shape=(784,)), Dense(128, activation='relu'), Dense(10, activation='softmax')])
    mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mlp_model.fit(X_train, y_train_cat, epochs=5, batch_size=512, verbose=0)

    cnn_model = Sequential([Input(shape=(28, 28, 1)), Conv2D(16, (3,3), activation='relu'), MaxPool2D((2,2)), Flatten(), Dense(10, activation='softmax')])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train_cat, epochs=5, batch_size=512, verbose=0)

    # --- 4. Тестирование на твоих фото ---
    log_stage("Тестирование на собственных фотографиях...")
    custom_results = []
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))

    for i in range(10):
        img_path = os.path.join(MY_NUMBERS_PATH, f"{i}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('L')

            # 1. Сначала уменьшаем — LANCZOS сгладит мелкие линии клеток
            img = img.resize((28, 28), Image.LANCZOS)

            # 2. Переводим в массив
            img_array = np.array(img, dtype=np.float32)

            # 3. Инвертируем (тёмная цифра на светлом фоне → светлая на тёмном)
            if img_array.mean() > 127:
                img_array = 255.0 - img_array

            # 4. Агрессивный порог — убираем ВСЁ кроме самой цифры
            #    Всё что тусклее порога → чёрный (0)
            img_array[img_array < 120] = 0

            # 5. Нормализация
            img_array = img_array / 255.0

            pred_mlp = np.argmax(mlp_model.predict(img_array.reshape(1, 784), verbose=0))
            pred_cnn = np.argmax(cnn_model.predict(img_array.reshape(1, 28, 28, 1), verbose=0))

            ax = axes.flatten()[i]
            ax.imshow(img_array, cmap='gray')
            ax.set_title(f"Real: {i}\nMLP: {pred_mlp}, CNN: {pred_cnn}")
            ax.axis('off')
            custom_results.append(f"Цифра {i}: MLP={pred_mlp}, CNN={pred_cnn}")

    # Матрицы ошибок и отчеты
    for name, model, data in [('MLP', mlp_model, X_test), ('CNN', cnn_model, X_test_cnn)]:
        pred = np.argmax(model.predict(data, verbose=0), axis=1)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
        plt.savefig(os.path.join(ASSETS_DIR, f'{name.lower()}_confusion.png'))
        plt.close()

    log_stage("Готово! Проверь assets/custom_test.png")

if __name__ == "__main__":
    run_lab()
