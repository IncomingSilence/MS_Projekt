"""Trainings- und Validierungslogik mit Zwei-Phasen-Transfer-Learning.

Phase A (Feature Extraction): Backbone eingefroren, nur der neue Kopf lernt.
Phase B (Fine-Tuning):        Backbone entfroren, kleine Lernrate fuer alle Schichten.

Aufruf als Skript (Beispiel):
    python -m src.train --data-root data/plantvillage --backbone resnet18 \
        --epochs-head 3 --epochs-finetune 5
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import torch
import torch.nn as nn

from .data import create_dataloaders
from .model import (
    build_model,
    save_checkpoint,
    trainable_parameters,
    unfreeze_all,
)


def get_device():
    # NVIDIA CUDA *oder* AMD ROCm (Linux): die ROCm-Builds melden sich als "cuda".
    if torch.cuda.is_available():
        return torch.device("cuda")
    # AMD/Intel-GPU unter Windows via DirectML (z. B. Radeon RX 5700 XT).
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    # Apple Silicon (auf dem Intel-Mac nicht verfuegbar, aber portabel).
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Fuehrt eine Epoche aus. Mit optimizer = Training, ohne = Validierung."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_correct = 0
    n_seen = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += (outputs.argmax(1) == targets).sum().item()
            n_seen += inputs.size(0)

    return running_loss / n_seen, running_correct / n_seen


def _train_phase(
    model, train_loader, val_loader, criterion, optimizer, device, epochs,
    phase_name, history, best,
):
    """Trainiert ueber 'epochs' Epochen und aktualisiert den Best-Checkpoint."""
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{phase_name}] Epoche {epoch}/{epochs} "
            f"| train_loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val_loss {val_loss:.4f} acc {val_acc:.3f} | {dt:.1f}s"
        )

        if val_acc > best["val_acc"]:
            best["val_acc"] = val_acc
            best["state_dict"] = copy.deepcopy(model.state_dict())

    return best


def train(
    data_root: str,
    backbone: str = "resnet18",
    epochs_head: int = 3,
    epochs_finetune: int = 5,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    out_path: str = "models/best_model.pth",
):
    """Komplettes Zwei-Phasen-Training. Speichert das beste Modell nach Val-Accuracy."""
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, class_names = create_dataloaders(
        data_root, batch_size=batch_size
    )
    print(f"{len(class_names)} Klassen gefunden.")

    model = build_model(
        num_classes=len(class_names), backbone=backbone, freeze_backbone=True
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best = {"val_acc": 0.0, "state_dict": copy.deepcopy(model.state_dict())}

    # --- Phase A: nur der neue Kopf ---
    if epochs_head > 0:
        optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr_head)
        best = _train_phase(
            model, train_loader, val_loader, criterion, optimizer, device,
            epochs_head, "Feature-Extraction", history, best,
        )

    # --- Phase B: gesamtes Netz fine-tunen ---
    if epochs_finetune > 0:
        unfreeze_all(model)
        optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr_finetune)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_finetune
        )
        for epoch in range(1, epochs_finetune + 1):
            best = _train_phase(
                model, train_loader, val_loader, criterion, optimizer, device,
                1, f"Fine-Tuning {epoch}/{epochs_finetune}", history, best,
            )
            scheduler.step()

    # Bestes Modell wiederherstellen und speichern.
    model.load_state_dict(best["state_dict"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_checkpoint(out_path, model, class_names, backbone)
    print(f"Bestes Modell (val_acc={best['val_acc']:.3f}) gespeichert: {out_path}")

    return model, class_names, history


def main():
    parser = argparse.ArgumentParser(description="PlantVillage Transfer Learning")
    parser.add_argument("--data-root", required=True, help="Pfad zum Datensatz")
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs-head", type=int, default=3)
    parser.add_argument("--epochs-finetune", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--out", default="models/best_model.pth")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        backbone=args.backbone,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
