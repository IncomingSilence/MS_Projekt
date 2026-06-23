"""Erzeugt Diagramme aus den Benchmark-Ergebnissen (results/*/metrics.json).

- results/accuracy_vs_time.png : Val-Accuracy gegen Trainingszeit (+ Inferenz-Tempo)
- results/val_acc_curves.png   : Val-Accuracy je Epoche pro Modell

Aufruf:
    python -m bench.plot
"""
from __future__ import annotations

import glob
import json

import matplotlib.pyplot as plt


def load_all():
    runs = []
    for path in sorted(glob.glob("results/*/metrics.json")):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def plot_accuracy_vs_time(runs):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in runs:
        x = r["training"]["total_seconds"] / 60.0           # Minuten
        y = r["evaluation"]["accuracy"] * 100.0             # Prozent
        size = r["model"]["total_params"] / 1e6 * 80        # Punktgroesse ~ Params
        ax.scatter(x, y, s=size, alpha=0.6, edgecolors="black")
        ax.annotate(
            f"{r['backbone']}\n{r['model']['total_params']/1e6:.1f}M Params\n"
            f"{r['inference_speed']['ms_per_image']:.2f} ms/Bild",
            (x, y), textcoords="offset points", xytext=(12, 0),
            va="center", fontsize=9,
        )
    ax.set_xlabel("Trainingszeit (Minuten)")
    ax.set_ylabel("Validierungs-Accuracy (%)")
    ax.set_title("Accuracy vs. Trainingszeit\n(Punktgröße ∝ Parameterzahl)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/accuracy_vs_time.png", dpi=120, bbox_inches="tight")
    print("results/accuracy_vs_time.png")


def plot_val_curves(runs):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in runs:
        accs, labels = [], []
        i = 0
        for ph in r["training"]["phases"]:
            for e in ph["epochs"]:
                i += 1
                accs.append(e["val_acc"] * 100.0)
                labels.append(i)
        ax.plot(labels, accs, marker="o", label=r["backbone"])
    ax.set_xlabel("Epoche (gesamt: Feature-Extraction + Fine-Tuning)")
    ax.set_ylabel("Validierungs-Accuracy (%)")
    ax.set_title("Trainingsverlauf: Val-Accuracy je Epoche")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/val_acc_curves.png", dpi=120, bbox_inches="tight")
    print("results/val_acc_curves.png")


def main():
    runs = load_all()
    if not runs:
        print("Keine results/*/metrics.json gefunden.")
        return
    plot_accuracy_vs_time(runs)
    plot_val_curves(runs)


if __name__ == "__main__":
    main()
