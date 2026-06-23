"""Benchmark-Harness: trainiert, evaluiert und testet ein Backbone einheitlich.

Pro Lauf entsteht unter ``results/<backbone>/``:
    metrics.json             alle Kennzahlen maschinenlesbar (fuer den Vergleich)
    training_log.txt         menschenlesbares Trainingsprotokoll
    classification_report.txt sklearn-Report pro Klasse
    confusion_matrix.png     normalisierte Confusion Matrix
Der Checkpoint landet unter ``models/<backbone>.pth``.

Aufruf:
    python -m bench.benchmark --backbone resnet18
    python -m bench.benchmark --backbone efficientnet_b0 --epochs-head 2 --epochs-finetune 3
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from datetime import datetime

import torch
import torch.nn as nn

from src.data import build_transforms, create_dataloaders, split_label
from src.evaluate import collect_predictions, plot_confusion_matrix
from src.model import (
    build_model,
    load_checkpoint,
    save_checkpoint,
    trainable_parameters,
    unfreeze_all,
)
from src.train import get_device, run_epoch

# Datenpfad mit train/valid (verschachtelt im Kaggle-Download).
DATA_ROOT = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TEST_DIR = "data/test/test"

# Dateinamen-Praefix der unabhaengigen Testbilder -> echte Klasse (Ground Truth).
TEST_PREFIX_TO_CLASS = {
    "AppleCedarRust": "Apple___Cedar_apple_rust",
    "AppleScab": "Apple___Apple_scab",
    "CornCommonRust": "Corn_(maize)___Common_rust_",
    "PotatoEarlyBlight": "Potato___Early_blight",
    "PotatoHealthy": "Potato___healthy",
    "TomatoEarlyBlight": "Tomato___Early_blight",
    "TomatoHealthy": "Tomato___healthy",
    "TomatoYellowCurlVirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


def _truth_from_filename(name: str) -> str | None:
    """Leitet die wahre Klasse aus dem Dateinamen ab (Ziffern + Endung entfernen)."""
    stem = os.path.splitext(name)[0]
    prefix = re.sub(r"\d+$", "", stem)
    return TEST_PREFIX_TO_CLASS.get(prefix)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_and_record(
    backbone, train_loader, val_loader, class_names, device,
    epochs_head, epochs_finetune, lr_head, lr_finetune, log_lines,
):
    """Zwei-Phasen-Training mit Erfassung von Loss/Acc/Zeit je Epoche."""
    model = build_model(
        num_classes=len(class_names), backbone=backbone, freeze_backbone=True
    ).to(device)
    total_params, head_trainable = count_parameters(model)
    criterion = nn.CrossEntropyLoss()

    phases = []
    best = {"val_acc": 0.0, "state_dict": copy.deepcopy(model.state_dict())}
    t_total = time.time()

    def _phase(name, optimizer, scheduler, n_epochs):
        epoch_records = []
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
            dt = time.time() - t0
            if scheduler is not None:
                scheduler.step()
            rec = {
                "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc, "seconds": dt,
            }
            epoch_records.append(rec)
            line = (f"[{name}] Epoche {epoch}/{n_epochs} | "
                    f"train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                    f"val_loss {va_loss:.4f} acc {va_acc:.3f} | {dt:.1f}s")
            print(line)
            log_lines.append(line)
            if va_acc > best["val_acc"]:
                best["val_acc"] = va_acc
                best["state_dict"] = copy.deepcopy(model.state_dict())
        phases.append({"name": name, "epochs": epoch_records})

    if epochs_head > 0:
        opt = torch.optim.Adam(trainable_parameters(model), lr=lr_head)
        _phase("feature_extraction", opt, None, epochs_head)

    if epochs_finetune > 0:
        unfreeze_all(model)
        opt = torch.optim.Adam(trainable_parameters(model), lr=lr_finetune)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_finetune)
        _phase("fine_tuning", opt, sched, epochs_finetune)

    total_seconds = time.time() - t_total
    model.load_state_dict(best["state_dict"])

    training = {
        "phases": phases,
        "total_seconds": total_seconds,
        "best_val_acc": best["val_acc"],
    }
    model_info = {"total_params": total_params, "head_trainable_params": head_trainable}
    return model, training, model_info


def evaluate_and_record(model, val_loader, class_names, device, out_dir):
    """Accuracy, Per-Klassen-Report (als dict) und Confusion-Matrix-PNG."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_true, y_pred = collect_predictions(model, val_loader, device)
    acc = float(accuracy_score(y_true, y_pred))

    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report_txt}\n")

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    per_class = {
        cls: {
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": report[cls]["support"],
        }
        for cls in class_names
    }

    fig = plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")

    return {
        "accuracy": acc,
        "macro_avg": {
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
        },
        "weighted_avg": {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        },
        "per_class": per_class,
        "confusion_matrix_png": cm_path.replace("\\", "/"),
    }


@torch.no_grad()
def measure_inference_speed(model, device, batch_size=32, n_batches=20, warmup=3):
    """Misst Inferenz-Durchsatz (Bilder/s) auf dem gegebenen Device."""
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(warmup):
        _ = model(x).cpu()  # .cpu() synchronisiert DirectML/CUDA
    t0 = time.time()
    for _ in range(n_batches):
        _ = model(x).cpu()
    dt = time.time() - t0
    images = batch_size * n_batches
    return {
        "device": str(device),
        "batch_size": batch_size,
        "images_per_sec": images / dt,
        "ms_per_image": 1000.0 * dt / images,
    }


@torch.no_grad()
def test_image_predictions(model, class_names, device):
    """Vorhersage auf den unabhaengigen Testbildern + Abgleich mit Dateinamen."""
    import torch.nn.functional as F
    from PIL import Image

    transform = build_transforms(train=False)
    model.eval()
    preds = []
    n_correct = n_labeled = 0
    if not os.path.isdir(TEST_DIR):
        return {"n": 0, "predictions": []}

    for name in sorted(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, name)
        if not os.path.isfile(path):
            continue
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu()
        idx = int(probs.argmax())
        pred = class_names[idx]
        truth = _truth_from_filename(name)
        correct = (truth == pred) if truth else None
        if truth is not None:
            n_labeled += 1
            n_correct += int(correct)
        plant, condition = split_label(pred)
        preds.append({
            "image": name,
            "true": truth,
            "pred": pred,
            "plant": plant,
            "condition": condition,
            "confidence": float(probs[idx]),
            "correct": correct,
        })

    return {
        "n": len(preds),
        "n_labeled": n_labeled,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_labeled) if n_labeled else None,
        "predictions": preds,
    }


def run(backbone, epochs_head, epochs_finetune, batch_size, lr_head, lr_finetune):
    device = get_device()
    print(f"=== Benchmark {backbone} === Device: {device}")

    out_dir = os.path.join("results", backbone)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_loader, val_loader, class_names = create_dataloaders(DATA_ROOT, batch_size=batch_size)
    print(f"{len(class_names)} Klassen | train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)}")

    log_lines = []
    model, training, model_info = train_and_record(
        backbone, train_loader, val_loader, class_names, device,
        epochs_head, epochs_finetune, lr_head, lr_finetune, log_lines,
    )

    ckpt_path = os.path.join("models", f"{backbone}.pth")
    save_checkpoint(ckpt_path, model, class_names, backbone)

    print("Evaluiere auf Validierungssplit ...")
    evaluation = evaluate_and_record(model, val_loader, class_names, device, out_dir)
    print(f"  Accuracy: {evaluation['accuracy']:.4f}")

    print("Messe Inferenzgeschwindigkeit ...")
    speed = measure_inference_speed(model, device, batch_size=batch_size)
    print(f"  {speed['images_per_sec']:.0f} Bilder/s ({speed['ms_per_image']:.2f} ms/Bild)")

    print("Teste unabhaengige Testbilder ...")
    test_res = test_image_predictions(model, class_names, device)
    if test_res["n_labeled"]:
        print(f"  {test_res['n_correct']}/{test_res['n_labeled']} korrekt")

    size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
    metrics = {
        "backbone": backbone,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "config": {
            "epochs_head": epochs_head, "epochs_finetune": epochs_finetune,
            "batch_size": batch_size, "lr_head": lr_head, "lr_finetune": lr_finetune,
            "image_size": 224,
        },
        "dataset": {
            "num_classes": len(class_names),
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
        },
        "model": {**model_info, "checkpoint": ckpt_path.replace("\\", "/"),
                  "size_mb": size_mb},
        "training": training,
        "evaluation": evaluation,
        "inference_speed": speed,
        "test_images": test_res,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"Ergebnisse geschrieben nach {out_dir}/")
    print("BENCHMARK_OK")
    return metrics


def main():
    p = argparse.ArgumentParser(description="Benchmark eines Backbones")
    p.add_argument("--backbone", required=True, choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--epochs-head", type=int, default=2)
    p.add_argument("--epochs-finetune", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    args = p.parse_args()
    run(args.backbone, args.epochs_head, args.epochs_finetune,
        args.batch_size, args.lr_head, args.lr_finetune)


if __name__ == "__main__":
    main()
