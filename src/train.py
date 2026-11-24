import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .dataset import make_loaders
from .models import build_model

HPARAMS = {
    "CNN":      dict(lr=0.00001, batch_size=64, epochs=10),
    "ResNet50": dict(lr=0.0001, batch_size=64, epochs=10),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, ys, probs = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item() * len(y)
        probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
    ys, probs = np.concatenate(ys), np.concatenate(probs)
    preds = (probs >= 0.5).astype(int)
    metrics = dict(
        loss=total_loss / len(ys),
        acc=accuracy_score(ys, preds),
        f1_macro=f1_score(ys, preds, average="macro"),
        f1_normal=f1_score(ys, preds, pos_label=0),
        f1_pneumonia=f1_score(ys, preds, pos_label=1),
        auc=roc_auc_score(ys, probs),
    )
    return metrics, ys, probs


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def run_one(name, seed, splits, device, workers, out_dir: Path):
    set_seed(seed)
    hp = HPARAMS[name]
    train_loader, val_loader, test_loader = make_loaders(splits, hp["batch_size"], seed, workers)

    model = build_model(name).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=hp["lr"])
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    t0 = time.time()
    for ep in range(1, hp["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val, _, _ = evaluate(model, val_loader, device, criterion)
        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=val["loss"], val_acc=val["acc"]))
        print(f"  [{name} seed={seed}] ep{ep:02d} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}")

        # guarda o checkpoint com menor loss de validação (o teste não é usado para escolher)
        if val["loss"] < best_val_loss:
            best_val_loss, best_epoch = val["loss"], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0

    model.load_state_dict(best_state)  # restaura o melhor checkpoint antes do teste

    metrics, ys, probs = evaluate(model, test_loader, device, criterion)
    metrics.update(model=name, seed=seed, train_time=train_time, best_epoch=best_epoch)

    # guarda histórico e probabilidades (permite plotar e retomar depois)
    (out_dir / "histories").mkdir(exist_ok=True)
    (out_dir / "probs").mkdir(exist_ok=True)
    with open(out_dir / "histories" / f"{name}_seed{seed}.json", "w") as f:
        json.dump(history, f)
    np.savez(out_dir / "probs" / f"{name}_seed{seed}.npz", y=ys, p=probs)
    return metrics