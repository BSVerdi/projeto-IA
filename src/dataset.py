from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["NORMAL", "PNEUMONIA"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SPLIT_SEED = 42


def collect_files(data_dir: Path):
    paths, labels = [], []

    for label, cls in enumerate(CLASSES):
        for p in sorted((data_dir / cls).glob("*")):
            if p.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                paths.append(str(p))
                labels.append(label)

    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem em {data_dir}/{{NORMAL,PNEUMONIA}}")

    return np.array(paths), np.array(labels)


def make_splits(paths, labels):
    p_train, p_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SPLIT_SEED)

    p_val, p_test, y_val, y_test = train_test_split(
        p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SPLIT_SEED)

    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


class XRayDataset(Dataset):
    def __init__(self, paths, labels, train: bool):
        self.paths, self.labels = paths, labels
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)] if train else []
        self.tf = transforms.Compose(
            [transforms.Resize((224, 224))] + aug +
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])


def make_loaders(splits, batch_size: int, seed: int, workers: int):
    (ptr, ytr), (pva, yva), (pte, yte) = splits

    g = torch.Generator().manual_seed(seed)

    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)

    train = DataLoader(XRayDataset(ptr, ytr, True), batch_size, shuffle=True, generator=g, **kw)
    val = DataLoader(XRayDataset(pva, yva, False), batch_size, **kw)
    test = DataLoader(XRayDataset(pte, yte, False), batch_size, **kw)

    return train, val, test