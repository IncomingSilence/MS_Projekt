"""Evaluation: Accuracy, Per-Klassen-Report und Confusion Matrix."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Sammelt alle Vorhersagen und wahren Labels eines DataLoaders."""
    model.eval()
    y_true, y_pred = [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(targets.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def evaluate(model, loader, class_names, device=None):
    """Gibt Accuracy aus und druckt den sklearn-Classification-Report.

    Returns
    -------
    accuracy, y_true, y_pred
    """
    from sklearn.metrics import accuracy_score, classification_report

    if device is None:
        device = next(model.parameters()).device

    y_true, y_pred = collect_predictions(model, loader, device)
    acc = accuracy_score(y_true, y_pred)
    print(f"Test-Accuracy: {acc:.4f}\n")
    print(classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    ))
    return acc, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, figsize=(12, 10)):
    """Zeichnet die Confusion Matrix. Gibt das matplotlib-Figure-Objekt zurueck."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Wahre Klasse",
        xlabel="Vorhergesagte Klasse",
        title="Confusion Matrix" + (" (normalisiert)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    return fig
