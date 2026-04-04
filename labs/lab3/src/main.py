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

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent
LAB3_DIR = BASE_DIR.parent
DATA_DIR = LAB3_DIR / "data"
ASSETS_DIR = LAB3_DIR / "assets"
WEIGHTS_DIR = ASSETS_DIR / "models"
FEATURE_MAPS_DIR = ASSETS_DIR / "feature_maps"

TRAIN_PATH = DATA_DIR / "train.csv"
MY_NUMBERS_PATH = DATA_DIR / "my_numbers"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_MAPS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitDataset(Dataset):
    def __init__(self, x_data, y_data=None, transform=None):
        self.x_data = x_data.values if isinstance(x_data, pd.DataFrame) else x_data
        self.y_data = y_data.values if hasattr(y_data, "values") else y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx].reshape(28, 28).astype(np.uint8)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        if self.y_data is None:
            return image

        label = int(self.y_data[idx])
        return image, label


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc_1 = nn.Linear(64 * 7 * 7, 128)
        self.fc_2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.fc_2(x)
        return x


def build_transform(use_aug=False):
    if use_aug:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.12, 0.12),
                scale=(0.9, 1.1),
                shear=10,
                fill=0
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=0),
            transforms.ToTensor(),
        ])

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


def prepare_dataloaders(batch_size=64, use_aug_train=False, use_aug_valid=False):
    print("Загрузка датасета для ЛР3...")
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

    train_ds = DigitDataset(x_train, y_train, transform=build_transform(use_aug_train))
    valid_ds = DigitDataset(x_valid, y_valid, transform=build_transform(use_aug_valid))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    with open(ASSETS_DIR / "dataset_info.txt", "w", encoding="utf-8") as f:
        f.write("Информация о данных ЛР3\n")
        f.write(f"Размер обучающей выборки: {x_train.shape}\n")
        f.write(f"Размер валидационной выборки: {x_valid.shape}\n")
        f.write(f"Количество классов: {len(sorted(target.unique()))}\n")
        f.write(f"Классы: {sorted(target.unique().tolist())}\n")

    return train_loader, valid_loader, x_valid, y_valid


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
            preds = torch.argmax(outputs, dim=1)
            correct_sum += (preds == labels).sum().item()
            object_sum += labels.size(0)

    return loss_sum / object_sum, correct_sum / object_sum


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


def evaluate_model(model, loader):
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            true_labels.extend(labels.numpy().tolist())
            pred_labels.extend(preds.tolist())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    acc = float((true_labels == pred_labels).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        average="macro",
        zero_division=0
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(10)))

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, title, file_name):
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
    plt.savefig(ASSETS_DIR / file_name)
    plt.close()


def preprocess_my_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    image_array = np.array(image, dtype=np.float32)

    if image_array.mean() > 127:
        image_array = 255 - image_array

    image_array = image_array / 255.0
    tensor = torch.tensor(image_array).unsqueeze(0)

    return tensor


def make_augmented_copy(tensor):
    aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.12, 0.12),
            scale=(0.9, 1.1),
            shear=10,
            fill=0
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0, fill=0),
        transforms.ToTensor(),
    ])
    return aug(tensor)


def predict_my_numbers(model, folder_path, add_augmented=False):
    results = []

    if not folder_path.exists():
        return results

    files = sorted([x for x in os.listdir(folder_path) if x.lower().endswith(".png")])

    model.eval()
    with torch.no_grad():
        for file_name in files:
            full_path = folder_path / file_name
            base_tensor = preprocess_my_image(full_path)

            base_pred = int(torch.argmax(model(base_tensor.unsqueeze(0).to(DEVICE)), dim=1).item())
            results.append((file_name, "обычное", base_pred))

            if add_augmented:
                aug_tensor = make_augmented_copy(base_tensor)
                aug_pred = int(torch.argmax(model(aug_tensor.unsqueeze(0).to(DEVICE)), dim=1).item())
                results.append((file_name, "аугментированное", aug_pred))

    return results


def save_my_numbers_results(title, rows, out_path):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(title + "\n")
        for file_name, mode_name, pred in rows:
            f.write(f"{file_name} | {mode_name} -> {pred}\n")
        f.write("\n")


def extract_feature_maps(model, image_tensor):
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(DEVICE)

        conv1 = F.relu(model.conv_1(x))
        pool1 = model.pool(conv1)

        conv2 = F.relu(model.conv_2(pool1))
        pool2 = model.pool(conv2)

    return {
        "conv1": conv1.squeeze(0).cpu(),
        "pool1": pool1.squeeze(0).cpu(),
        "conv2": conv2.squeeze(0).cpu(),
        "pool2": pool2.squeeze(0).cpu(),
    }


def save_feature_grid(feature_tensor, file_path, cols=8):
    channels = feature_tensor.shape[0]
    rows = int(np.ceil(channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.7))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")

        if idx >= channels:
            continue

        fmap = feature_tensor[idx].numpy()
        fmin, fmax = float(fmap.min()), float(fmap.max())

        if fmax > fmin:
            fmap = (fmap - fmin) / (fmax - fmin)
        else:
            fmap = np.zeros_like(fmap)

        ax.imshow(fmap, cmap="magma")
        ax.set_title(f"#{idx}", fontsize=7)

    plt.tight_layout()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_path)
    plt.close()


def save_feature_maps_for_folder(model, folder_path, root_dir):
    if not folder_path.exists():
        return

    files = sorted([x for x in os.listdir(folder_path) if x.lower().endswith(".png")])

    for file_name in files:
        full_path = folder_path / file_name
        tensor = preprocess_my_image(full_path)
        maps = extract_feature_maps(model, tensor)

        target_dir = root_dir / Path(file_name).stem
        target_dir.mkdir(parents=True, exist_ok=True)

        for layer_name, fmap in maps.items():
            save_feature_grid(fmap, target_dir / f"{layer_name}.png")


def write_summary(report_data):
    with open(ASSETS_DIR / "lab3_summary.txt", "w", encoding="utf-8") as f:
        f.write("Лабораторная работа 3\n\n")

        for block_title, block_metrics in report_data.items():
            f.write(block_title + "\n")
            for metric_name, metric_value in block_metrics.items():
                if metric_name == "confusion_matrix":
                    continue
                f.write(f"{metric_name}: {metric_value:.4f}\n")
            f.write("\n")


def main():
    print(f"Используемое устройство: {DEVICE}")

    # 1. baseline CNN без аугментации
    train_loader_clean, valid_loader_clean, x_valid, y_valid = prepare_dataloaders(
        batch_size=64,
        use_aug_train=False,
        use_aug_valid=False
    )

    baseline_cnn = CNNModel().to(DEVICE)
    baseline_acc, baseline_path = train_model(
        baseline_cnn,
        train_loader_clean,
        valid_loader_clean,
        model_label="Baseline CNN",
        weight_name="baseline_cnn_best.pth",
        plot_name="baseline_cnn_curves.png",
        epochs=10,
        lr=0.001
    )

    baseline_cnn.load_state_dict(torch.load(baseline_path, map_location=DEVICE))

    baseline_clean_metrics = evaluate_model(baseline_cnn, valid_loader_clean)

    _, valid_loader_aug, _, _ = prepare_dataloaders(
        batch_size=64,
        use_aug_train=False,
        use_aug_valid=True
    )

    baseline_aug_metrics = evaluate_model(baseline_cnn, valid_loader_aug)

    save_confusion_matrix(
        baseline_aug_metrics["confusion_matrix"],
        "Baseline CNN на аугментированной валидации",
        "confusion_matrix_baseline_extended.png"
    )

    baseline_my_numbers = predict_my_numbers(baseline_cnn, MY_NUMBERS_PATH, add_augmented=True)

    baseline_feature_root = FEATURE_MAPS_DIR / "baseline"
    save_feature_maps_for_folder(baseline_cnn, MY_NUMBERS_PATH, baseline_feature_root)

    # 2. CNN с аугментированным обучением
    train_loader_aug, valid_loader_clean_2, _, _ = prepare_dataloaders(
        batch_size=64,
        use_aug_train=True,
        use_aug_valid=False
    )

    augmented_cnn = CNNModel().to(DEVICE)
    augmented_acc, augmented_path = train_model(
        augmented_cnn,
        train_loader_aug,
        valid_loader_clean_2,
        model_label="CNN с аугментацией",
        weight_name="augmented_cnn_best.pth",
        plot_name="augmented_cnn_curves.png",
        epochs=10,
        lr=0.001
    )

    augmented_cnn.load_state_dict(torch.load(augmented_path, map_location=DEVICE))

    augmented_clean_metrics = evaluate_model(augmented_cnn, valid_loader_clean_2)
    augmented_aug_metrics = evaluate_model(augmented_cnn, valid_loader_aug)

    save_confusion_matrix(
        augmented_aug_metrics["confusion_matrix"],
        "CNN с аугментацией на аугментированной валидации",
        "confusion_matrix_augmented_extended.png"
    )

    augmented_my_numbers = predict_my_numbers(augmented_cnn, MY_NUMBERS_PATH, add_augmented=True)

    augmented_feature_root = FEATURE_MAPS_DIR / "augmented"
    save_feature_maps_for_folder(augmented_cnn, MY_NUMBERS_PATH, augmented_feature_root)

    my_results_path = ASSETS_DIR / "my_numbers_comparison.txt"
    if my_results_path.exists():
        os.remove(my_results_path)

    save_my_numbers_results("Baseline CNN", baseline_my_numbers, my_results_path)
    save_my_numbers_results("CNN с аугментацией", augmented_my_numbers, my_results_path)

    report_data = {
        "Baseline CNN на обычной валидации": baseline_clean_metrics,
        "Baseline CNN на аугментированной валидации": baseline_aug_metrics,
        "CNN с аугментацией на обычной валидации": augmented_clean_metrics,
        "CNN с аугментацией на аугментированной валидации": augmented_aug_metrics,
    }

    write_summary(report_data)

    print("\nЛабораторная работа 3 завершена.")
    print("Все результаты сохранены в папке lab3/assets/.")


if __name__ == "__main__":
    main()
