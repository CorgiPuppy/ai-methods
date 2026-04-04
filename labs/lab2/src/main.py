import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
LAB2_DIR = BASE_DIR.parent
DATA_DIR = LAB2_DIR / "data"
ASSETS_DIR = LAB2_DIR / "assets"
WEIGHTS_DIR = ASSETS_DIR / "models"

TRAIN_PATH = DATA_DIR / "train.csv"
MY_NUMBERS_PATH = DATA_DIR / "my_numbers"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitDataset(Dataset):
    def __init__(self, x_data, y_data=None):
        self.x_data = x_data.values if isinstance(x_data, pd.DataFrame) else x_data
        self.y_data = y_data.values if hasattr(y_data, "values") else y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx].reshape(28, 28).astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)

        if self.y_data is None:
            return image

        label = int(self.y_data[idx])
        return image, label


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 320),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Архитектура намеренно отличается от прошлой версии:
        # 3 свёртки вместо 2, другие числа каналов
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 28x28 -> conv1 -> 28x28
        # -> conv2 -> 28x28
        # -> pool -> 14x14
        # -> conv3 -> 14x14
        # -> pool -> 7x7
        self.fc_1 = nn.Linear(64 * 7 * 7, 96)
        self.fc_2 = nn.Linear(96, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.pool(x)

        x = F.relu(self.conv_3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.fc_2(x)
        return x


def prepare_data(batch_size=64):
    print("Загрузка датасета...")
    df = pd.read_csv(TRAIN_PATH)

    features = df.drop("label", axis=1)
    target = df["label"]

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target
    )

    train_ds = DigitDataset(x_train, y_train)
    valid_ds = DigitDataset(x_valid, y_valid)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    with open(ASSETS_DIR / "dataset_info.txt", "w", encoding="utf-8") as f:
        f.write("Информация о данных\n")
        f.write(f"Размер обучающей выборки: {x_train.shape}\n")
        f.write(f"Размер валидационной выборки: {x_valid.shape}\n")
        f.write(f"Количество классов: {len(sorted(target.unique()))}\n")
        f.write(f"Классы: {sorted(target.unique().tolist())}\n")

    return train_loader, valid_loader


def one_pass(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None

    if train_mode:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    correct_sum = 0
    object_sum = 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if train_mode:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * images.size(0)
            predicts = torch.argmax(outputs, dim=1)
            correct_sum += (predicts == labels).sum().item()
            object_sum += labels.size(0)

    mean_loss = loss_sum / object_sum
    mean_acc = correct_sum / object_sum

    return mean_loss, mean_acc


def collect_predictions(model, loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(y_true, y_pred, title, out_name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.xticks(range(10))
    plt.yticks(range(10))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / out_name)
    plt.close()


def draw_training_curves(history, model_label, file_name):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Обучение")
    plt.plot(epochs, history["valid_loss"], label="Валидация")
    plt.title(f"{model_label}: функция потерь")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Обучение")
    plt.plot(epochs, history["valid_acc"], label="Валидация")
    plt.title(f"{model_label}: точность")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / file_name)
    plt.close()


def train_model(model, train_loader, valid_loader, model_label, weight_name, plot_name, epochs=10, lr=0.001):
    print(f"\nНачало обучения модели: {model_label}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    best_acc = 0.0
    best_path = WEIGHTS_DIR / weight_name

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = one_pass(model, train_loader, criterion, optimizer=optimizer)
        valid_loss, valid_acc = one_pass(model, valid_loader, criterion, optimizer=None)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(
            f"{model_label} | эпоха {epoch}/{epochs} | "
            f"loss train={train_loss:.4f}, acc train={train_acc:.4f} | "
            f"loss val={valid_loss:.4f}, acc val={valid_acc:.4f}"
        )

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), best_path)

    draw_training_curves(history, model_label, plot_name)

    return best_acc, best_path


def preprocess_my_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    image_array = np.array(image, dtype=np.float32)

    if image_array.mean() > 127:
        image_array = 255 - image_array

    image_array = image_array / 255.0
    tensor = torch.tensor(image_array).unsqueeze(0)

    return tensor


def predict_folder(model, folder_path):
    results = {}

    if not folder_path.exists():
        return results

    files = sorted([x for x in os.listdir(folder_path) if x.lower().endswith(".png")])

    model.eval()
    with torch.no_grad():
        for file_name in files:
            full_path = folder_path / file_name
            image_tensor = preprocess_my_image(full_path).unsqueeze(0).to(DEVICE)
            logits = model(image_tensor)
            pred = int(torch.argmax(logits, dim=1).item())
            results[file_name] = pred

    return results


def save_predictions_block(title, predictions, out_file):
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(title + "\n")
        for name, pred in predictions.items():
            f.write(f"{name} -> {pred}\n")
        f.write("\n")


def save_my_numbers_grid(folder_path, out_name="my_numbers_grid.png"):
    if not folder_path.exists():
        return

    image_files = sorted([x for x in os.listdir(folder_path) if x.lower().endswith(".png")])

    if len(image_files) == 0:
        return

    count = len(image_files)
    cols = 5
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")

        if idx >= count:
            continue

        file_name = image_files[idx]
        image = Image.open(folder_path / file_name).convert("L")
        ax.imshow(image, cmap="gray")
        ax.set_title(file_name, fontsize=10)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / out_name)
    plt.close()

def save_prediction_grid(folder_path, mlp_predictions, cnn_predictions, out_name="my_numbers_predictions_grid.png"):
    if not folder_path.exists():
        return

    image_files = sorted([x for x in os.listdir(folder_path) if x.lower().endswith(".png")])

    if len(image_files) == 0:
        return

    count = len(image_files)
    cols = 5
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 3.2))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")

        if idx >= count:
            continue

        file_name = image_files[idx]
        image = Image.open(folder_path / file_name).convert("L")

        mlp_pred = mlp_predictions.get(file_name, "?")
        cnn_pred = cnn_predictions.get(file_name, "?")

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"{file_name}\nMLP: {mlp_pred}\nCNN: {cnn_pred}",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / out_name, dpi=180)
    plt.close()

def main():
    print(f"Используемое устройство: {DEVICE}")

    train_loader, valid_loader = prepare_data(batch_size=64)

    mlp_model = MLPModel().to(DEVICE)
    cnn_model = CNNModel().to(DEVICE)

    mlp_acc, mlp_best_path = train_model(
        model=mlp_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model_label="MLP",
        weight_name="mlp_best.pth",
        plot_name="mlp_curves.png",
        epochs=10,
        lr=0.001
    )

    cnn_acc, cnn_best_path = train_model(
        model=cnn_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model_label="CNN",
        weight_name="cnn_best.pth",
        plot_name="cnn_curves.png",
        epochs=10,
        lr=0.001
    )

    # Загружаем лучшие версии
    mlp_model.load_state_dict(torch.load(mlp_best_path, map_location=DEVICE))
    cnn_model.load_state_dict(torch.load(cnn_best_path, map_location=DEVICE))

    # confusion matrix
    mlp_true, mlp_pred = collect_predictions(mlp_model, valid_loader)
    cnn_true, cnn_pred = collect_predictions(cnn_model, valid_loader)

    save_confusion_matrix(
        mlp_true,
        mlp_pred,
        "Матрица ошибок MLP",
        "mlp_confusion_matrix.png"
    )

    save_confusion_matrix(
        cnn_true,
        cnn_pred,
        "Матрица ошибок CNN",
        "cnn_confusion_matrix.png"
    )

    # summary
    with open(ASSETS_DIR / "training_summary.txt", "w", encoding="utf-8") as f:
        f.write("Итоговые результаты обучения\n")
        f.write(f"Лучшая точность MLP на валидации: {mlp_acc:.4f}\n")
        f.write(f"Лучшая точность CNN на валидации: {cnn_acc:.4f}\n")
        f.write(f"Файл весов MLP: {mlp_best_path}\n")
        f.write(f"Файл весов CNN: {cnn_best_path}\n")

    print("\nПроверка моделей на собственных изображениях...")

    predictions_file = ASSETS_DIR / "my_digits_predictions.txt"
    if predictions_file.exists():
        os.remove(predictions_file)

    mlp_predictions = predict_folder(mlp_model, MY_NUMBERS_PATH)
    cnn_predictions = predict_folder(cnn_model, MY_NUMBERS_PATH)

    print("\nПредсказания MLP:")
    for name, pred in mlp_predictions.items():
        print(f"{name}: {pred}")

    print("\nПредсказания CNN:")
    for name, pred in cnn_predictions.items():
        print(f"{name}: {pred}")

    save_predictions_block("Предсказания MLP", mlp_predictions, predictions_file)
    save_predictions_block("Предсказания CNN", cnn_predictions, predictions_file)

    save_my_numbers_grid(MY_NUMBERS_PATH, out_name="my_numbers_grid.png")

    save_prediction_grid(
        MY_NUMBERS_PATH,
        mlp_predictions,
        cnn_predictions,
        out_name="my_numbers_predictions_grid.png"
    )

    print("\nРабота завершена. Результаты сохранены в папке lab2/assets/.")


if __name__ == "__main__":
    main()
