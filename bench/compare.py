"""Liest alle results/<run-name>/metrics.json und erzeugt results/BENCHMARK.md.

Die Laeufe werden nach ``task`` gruppiert, damit verschiedene Aufgaben
(z.B. Crop-Krankheiten vs. Zimmerpflanzen-Arten) nicht in einer Tabelle
vermischt werden. Pro Task entsteht ein eigener Abschnitt.

Aufruf:
    python -m bench.compare
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import OrderedDict


HISTORY_PATH = "results/history.jsonl"


def load_all():
    runs = []
    for path in sorted(glob.glob("results/*/metrics.json")):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def load_history():
    """Liest results/history.jsonl (eine JSON-Zeile pro Lauf, append-only)."""
    if not os.path.exists(HISTORY_PATH):
        return []
    rows = []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fmt_secs(s):
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


def _label(run):
    """Eindeutiges Label eines Laufs (Run-Name, faellt auf Backbone zurueck)."""
    return run.get("run_name") or run.get("backbone", "?")


def _task(run):
    return run.get("task", "crop_disease")


def _slug(task):
    """Dateinamenfreundlicher Task-Name (muss zu bench/plot.py passen)."""
    return re.sub(r"[^0-9A-Za-z_-]+", "_", task)


def _worst_classes(run, k=3):
    """Die k Klassen mit dem niedrigsten F1-Score."""
    per = run["evaluation"]["per_class"]
    ranked = sorted(per.items(), key=lambda kv: kv[1]["f1"])
    return ranked[:k]


def group_by_task(runs):
    """Gruppiert die Laeufe nach task; Reihenfolge nach erstem Auftreten."""
    groups = OrderedDict()
    for r in runs:
        groups.setdefault(_task(r), []).append(r)
    return groups


def build_task_section(task, runs):
    """Erzeugt den Markdown-Abschnitt fuer eine Task-Gruppe."""
    ds = runs[0]["dataset"]
    dataset_name = runs[0].get("dataset_name")
    # Tausenderpunkte nur fuer die Zahlen, nicht fuer den ganzen Satz.
    n_train = f"{ds['n_train']:,}".replace(",", ".")
    n_val = f"{ds['n_val']:,}".replace(",", ".")
    # Alte metrics.json haben kein dataset_name -> Name-Prefix weglassen.
    name_prefix = f"{dataset_name} — " if dataset_name else ""
    lines = [
        f"# Task: {task}",
        "",
        f"**Datensatz:** {name_prefix}"
        f"{ds['num_classes']} Klassen, {n_train} Train- / {n_val} Val-Bilder.",
        f"**Hardware/Device:** {runs[0]['device']}",
        "",
        "## Überblick",
        "",
        "| Modell | Val-Accuracy | Macro-F1 | Params | Modellgröße | Train-Zeit | Inferenz | Testbilder |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        ev = r["evaluation"]
        sp = r["inference_speed"]
        ti = r["test_images"]
        test_str = (f"{ti['n_correct']}/{ti['n_labeled']}"
                    if ti.get("n_labeled") else
                    (f"{ti['n']} (ohne GT)" if ti.get("n") else "—"))
        lines.append(
            f"| **{_label(r)}** "
            f"| {ev['accuracy']*100:.2f}% "
            f"| {ev['macro_avg']['f1']:.4f} "
            f"| {r['model']['total_params']/1e6:.1f}M "
            f"| {r['model']['size_mb']:.1f} MB "
            f"| {_fmt_secs(r['training']['total_seconds'])} "
            f"| {sp['ms_per_image']:.2f} ms/Bild "
            f"| {test_str} |"
        )

    lines += ["", "## Konfiguration", ""]
    for r in runs:
        c = r["config"]
        lines.append(
            f"- **{_label(r)}** ({r.get('backbone', '?')}): {c['epochs_head']} Kopf- + "
            f"{c['epochs_finetune']} Fine-Tuning-Epochen, batch={c['batch_size']}, "
            f"lr_head={c['lr_head']}, lr_finetune={c['lr_finetune']}"
        )

    lines += ["", "## Trainingsverlauf (Val-Accuracy je Epoche)", ""]
    for r in runs:
        accs = []
        for ph in r["training"]["phases"]:
            accs += [f"{e['val_acc']*100:.1f}%" for e in ph["epochs"]]
        lines.append(f"- **{_label(r)}**: " + " → ".join(accs))

    lines += ["", "## Schwächste Klassen (niedrigster F1)", ""]
    for r in runs:
        worst = _worst_classes(r, k=3)
        parts = [f"{cls} ({m['f1']:.3f})" for cls, m in worst]
        lines.append(f"- **{_label(r)}**: " + ", ".join(parts))

    lines += ["", "## Testbild-Vorhersagen (unabhängige Fotos)", ""]
    for r in runs:
        ti = r["test_images"]
        if not ti.get("predictions"):
            continue
        lines += [f"### {_label(r)}", "",
                  "| Bild | Wahrheit | Vorhersage | Konfidenz | ✓ |",
                  "|---|---|---|---|---|"]
        for p in ti["predictions"]:
            mark = "✅" if p["correct"] else ("❌" if p["correct"] is False else "—")
            lines.append(
                f"| {p['image']} | {p['true'] or '—'} | {p['pred']} "
                f"| {p['confidence']*100:.1f}% | {mark} |"
            )
        lines.append("")

    # Diagramme dieser Task verlinken (sofern bench/plot.py gelaufen ist).
    slug = _slug(task)
    plots = [("Accuracy vs. Trainingszeit", f"{slug}_accuracy_vs_time.png"),
             ("Val-Accuracy je Epoche", f"{slug}_val_acc_curves.png")]
    existing = [(t, f) for t, f in plots if os.path.exists(os.path.join("results", f))]
    if existing:
        lines += ["## Diagramme", ""]
        for title, f in existing:
            lines.append(f"**{title}**\n\n![{title}]({f})\n")

    # Confusion-Matrizen verlinken.
    lines += ["## Confusion-Matrizen", ""]
    for r in runs:
        cm = r["evaluation"].get("confusion_matrix_png")
        if cm:
            rel = os.path.relpath(cm, "results").replace("\\", "/")
            lines.append(f"- **{_label(r)}**: ![{_label(r)}]({rel})")
    lines.append("")
    return lines


def build_history_section(history):
    """Chronologische Tabelle aller je gelaufenen Benchmarks, gruppiert nach task.

    Speist sich aus dem append-only Log (results/history.jsonl) und bleibt damit
    auch erhalten, wenn ein results/<run-name>/ spaeter ueberschrieben wird.
    """
    if not history:
        return []
    lines = ["# Historie (alle Läufe, chronologisch)", "",
             "_Aus `results/history.jsonl` — zeigt den Effekt von Modell- und "
             "Hyperparameter-Änderungen über die Zeit._", ""]
    groups = OrderedDict()
    for row in history:
        groups.setdefault(row.get("task", "?"), []).append(row)

    for task, rows in groups.items():
        rows = sorted(rows, key=lambda r: r.get("timestamp", ""))
        lines += [
            f"## {task}", "",
            "| Zeitpunkt | Run | Backbone | Epochen (K+F) | Batch | lr_head | lr_ft "
            "| Sampler | Val-Acc | Macro-F1 | Train-Zeit | Testbilder |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            ts = r.get("timestamp", "—").replace("T", " ")
            test_str = (f"{r['test_n_correct']}/{r['test_n_labeled']}"
                        if r.get("test_n_labeled") else "—")
            sampler = "weighted" if r.get("weighted_sampler") else "—"
            lines.append(
                f"| {ts} "
                f"| {r.get('run_name', '?')} "
                f"| {r.get('backbone', '?')} "
                f"| {r.get('epochs_head', '?')}+{r.get('epochs_finetune', '?')} "
                f"| {r.get('batch_size', '?')} "
                f"| {r.get('lr_head', '?')} "
                f"| {r.get('lr_finetune', '?')} "
                f"| {sampler} "
                f"| {r.get('val_accuracy', 0)*100:.2f}% "
                f"| {r.get('macro_f1', 0):.4f} "
                f"| {_fmt_secs(r.get('train_seconds', 0))} "
                f"| {test_str} |"
            )
        lines.append("")
    lines += ["---", ""]
    return lines


def build_markdown(runs, history=None):
    if not runs:
        return "Keine Ergebnisse gefunden (results/*/metrics.json fehlt)."

    groups = group_by_task(runs)
    lines = ["# Benchmark: Modellvergleich", ""]
    if len(groups) > 1:
        lines += ["**Tasks:** " + ", ".join(groups.keys()), ""]

    for task, task_runs in groups.items():
        lines += build_task_section(task, task_runs)
        lines += ["---", ""]

    lines += build_history_section(history or [])

    return "\n".join(lines)


def main():
    runs = load_all()
    history = load_history()
    md = build_markdown(runs, history)
    os.makedirs("results", exist_ok=True)
    out = "results/BENCHMARK.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"{len(runs)} Lauf/Läufe gefunden -> {out}")


if __name__ == "__main__":
    main()
