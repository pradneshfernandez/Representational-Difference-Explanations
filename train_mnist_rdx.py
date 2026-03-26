from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


SEED = 42
TARGET_DIGITS = {3, 5, 8}
TRAIN_PER_CLASS = 500
TEST_PER_CLASS = 300
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FilteredMNIST(Dataset):
    def __init__(self, base_dataset: Dataset, selected_indices: list[int]):
        self.base_dataset = base_dataset
        self.selected_indices = selected_indices

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[self.selected_indices[idx]]
        mapped = {3: 0, 5: 1, 8: 2}[int(y)]
        return x, mapped
    
def select_balanced_indices(targets, digits: set[int], per_class: int) -> list[int]:
    buckets = {d: [] for d in digits}
    for idx, y in enumerate(targets):
        y = int(y)
        if y in digits and len(buckets[y]) < per_class:
            buckets[y].append(idx)
        if all(len(v) >= per_class for v in buckets.values()):
            break

    selected = []
    for d in sorted(digits):
        selected.extend(buckets[d])
    return selected
class SmallCNN(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.embed = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.embed(x)
        logits = self.classifier(z)
        return logits

    def extract_embeddings(self, x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.embed(x)
        return z
@dataclass
class TrainOutputs:
    model_epoch_1: SmallCNN
    model_epoch_5: SmallCNN
    train_loader: DataLoader
    test_loader: DataLoader


def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])

    train_base = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_idx = select_balanced_indices(train_base.targets, TARGET_DIGITS, TRAIN_PER_CLASS)
    test_idx = select_balanced_indices(test_base.targets, TARGET_DIGITS, TEST_PER_CLASS)

    train_ds = FilteredMNIST(train_base, train_idx)
    test_ds = FilteredMNIST(test_base, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def clone_model(model: SmallCNN) -> SmallCNN:
    cloned = SmallCNN(embedding_dim=64, num_classes=3)
    cloned.load_state_dict(model.state_dict())
    return cloned
def train_models() -> TrainOutputs:
    set_seed()
    train_loader, test_loader = build_dataloaders()

    model = SmallCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_1_model = None
    epoch_5_model = None

    for epoch in range(1, 6):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch} | loss={running_loss / len(train_loader):.4f} | test_acc={acc:.4f}")

        if epoch == 1:
            epoch_1_model = clone_model(model).to("cpu")
        if epoch == 5:
            epoch_5_model = clone_model(model).to("cpu")

    assert epoch_1_model is not None and epoch_5_model is not None
    return TrainOutputs(
        model_epoch_1=epoch_1_model,
        model_epoch_5=epoch_5_model,
        train_loader=train_loader,
        test_loader=test_loader,
    )
def extract_embeddings(model: SmallCNN, loader: DataLoader, save_dir: Path, prefix: str) -> None:
    model = model.to(DEVICE)
    model.eval()

    all_embeddings = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for x, y in loader:
            z = model.extract_embeddings(x.to(DEVICE)).cpu().numpy()
            all_embeddings.append(z)
            all_labels.append(y.numpy())
            all_images.append(x.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    images = np.concatenate(all_images, axis=0)

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"{prefix}_embeddings.npy", embeddings)
    np.save(save_dir / f"{prefix}_labels.npy", labels)
    np.save(save_dir / f"{prefix}_images.npy", images)

    print(f"Saved {prefix} embeddings: {embeddings.shape}")


if __name__ == "__main__":
    outputs = train_models()
    out_dir = Path("data/mnist_rdx")

    extract_embeddings(outputs.model_epoch_1, outputs.test_loader, out_dir, "model_epoch_1")
    extract_embeddings(outputs.model_epoch_5, outputs.test_loader, out_dir, "model_epoch_5")