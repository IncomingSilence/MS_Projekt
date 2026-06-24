"""Erzeugt Diagramme aus den Benchmark-Ergebnissen (results/*/metrics.json).

Die Laeufe werden nach ``task`` getrennt, damit verschiedene Aufgaben (z.B.
Crop-Krankheiten vs. Zimmerpflanzen-Arten) nicht in einem Diagramm vermischt
werden. Pro Task entstehen:
    results/<task>_accuracy_vs_time.png : Val-Accuracy gegen Trainingszeit
    results/<task>_val_acc_curves.png   : Val-Accuracy je Epoche pro Lauf

Aufruf:
    python -m bench.plot
"""
from __future__ import annotations

import glob
import json
import re
from collections import OrderedDict

import matplotlib.pyplot as plt


def load_all():
    runs = []
    for path in sorted(glob.glob("results/*/metrics.json")):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def _label(run):
    return run.get("run_name") or run.get("backbone", "?")


def _task(run):
    return run.get("task", "crop_disease")


def _slug(task):
    """Dateinamenfreundlicher Task-Name."""
    return re.sub(r"[^0-9A-Za-z_-]+", "_", task)


def group_by_task(runs):
    groups = OrderedDict()
    for r in runs:
        groups.setdefault(_task(r), []).append(r)
    return groups


def plot_accuracy_vs_time(runs, task, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in runs:
        x = r["training"]["total_seconds"] / 60.0           # Minuten
        y = r["evaluation"]["accuracy"] * 100.0             # Prozent
        size = r["model"]["total_params"] / 1e6 * 80        # Punktgroesse ~ Params
        ax.scatter(x, y, s=size, alpha=0.6, edgecolors="black")
        ax.annotate(
            f"{_label(r)}\n{r['model']['total_params']/1e6:.1f}M Params\n"
            f"{r['inference_speed']['ms_per_image']:.2f} ms/Bild",
            (x, y), textcoords="offset points", xytext=(12, 0),
            va="center", fontsize=9,
        )
    ax.set_xlabel("Trainingszeit (Minuten)")
    ax.set_ylabel("Validierungs-Accuracy (%)")
    ax.set_title(f"Accuracy vs. Trainingszeit — {task}\n(Punktgröße ∝ Parameterzahl)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def plot_val_curves(runs, task, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in runs:
        accs, labels = [], []
        i = 0
        for ph in r["training"]["phases"]:
            for e in ph["epochs"]:
                i += 1
                accs.append(e["val_acc"] * 100.0)
                labels.append(i)
        ax.plot(labels, accs, marker="o", label=_label(r))
    ax.set_xlabel("Epoche (gesamt: Feature-Extraction + Fine-Tuning)")
    ax.set_ylabel("Validierungs-Accuracy (%)")
    ax.set_title(f"Trainingsverlauf: Val-Accuracy je Epoche — {task}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def main():
    runs = load_all()
    if not runs:
        print("Keine results/*/metrics.json gefunden.")
        return
    for task, task_runs in group_by_task(runs).items():
        slug = _slug(task)
        plot_accuracy_vs_time(task_runs, task, f"results/{slug}_accuracy_vs_time.png")
        plot_val_curves(task_runs, task, f"results/{slug}_val_acc_curves.png")


if __name__ == "__main__":
    main()
