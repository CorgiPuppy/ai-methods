import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'diamonds.csv')
ASSETS_DIR = os.path.join(BASE_DIR, '..', 'assets')

os.makedirs(ASSETS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Распределение цен на бриллианты')
plt.savefig(os.path.join(ASSETS_DIR, 'price_dist.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(df['carat'], df['price'], alpha=0.3)
plt.title('Зависимость цены от веса')
plt.xlabel('Вес')
plt.ylabel('Цена')
plt.savefig(os.path.join(ASSETS_DIR, 'price_carat.png'))
plt.close()

plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Корреляция признаков')
plt.savefig(os.path.join(ASSETS_DIR, 'heatmap.png'))
plt.close()

le = LabelEncoder()
for col in ['cut', 'color', 'clarity']:
    df[col] = le.fit_transform(df[col])
    
X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_single = Sequential([Dense(1, input_dim=X_train.shape[1])])
model_single.compile(optimizer='adam', loss='mean_squared_error')
model_single.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
mse_s = model_single.evaluate(X_test, y_test, verbose=0)

model_multi = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(64, activation='relu'),
    Dense(1)
])
model_multi.compile(optimizer='adam', loss='mean_squared_error')
history = model_multi.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
mse_m = model_multi.evaluate(X_test, y_test, verbose=0)

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('MLP Training Process')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig(os.path.join(ASSETS_DIR, 'training_plot.png'))
plt.close()
