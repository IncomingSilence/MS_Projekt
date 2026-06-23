"""Liest alle results/<backbone>/metrics.json und erzeugt results/BENCHMARK.md.

Aufruf:
    python -m bench.compare
"""
from __future__ import annotations

import glob
import json
import os


def load_all():
    runs = []
    for path in sorted(glob.glob("results/*/metrics.json")):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def _fmt_secs(s):
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


def _worst_classes(run, k=3):
    """Die k Klassen mit dem niedrigsten F1-Score."""
    per = run["evaluation"]["per_class"]
    ranked = sorted(per.items(), key=lambda kv: kv[1]["f1"])
    return ranked[:k]


def build_markdown(runs):
    lines = ["# Benchmark: Modellvergleich", ""]
    if not runs:
        return "Keine Ergebnisse gefunden (results/*/metrics.json fehlt)."

    ds = runs[0]["dataset"]
    # Tausenderpunkte nur fuer die Zahlen, nicht fuer den ganzen Satz.
    n_train = f"{ds['n_train']:,}".replace(",", ".")
    n_val = f"{ds['n_val']:,}".replace(",", ".")
    lines += [
        f"**Datensatz:** PlantVillage (New Plant Diseases, augmentiert) — "
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
                    if ti.get("n_labeled") else "—")
        lines.append(
            f"| **{r['backbone']}** "
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
            f"- **{r['backbone']}**: {c['epochs_head']} Kopf- + "
            f"{c['epochs_finetune']} Fine-Tuning-Epochen, batch={c['batch_size']}, "
            f"lr_head={c['lr_head']}, lr_finetune={c['lr_finetune']}"
        )

    lines += ["", "## Trainingsverlauf (Val-Accuracy je Epoche)", ""]
    for r in runs:
        accs = []
        for ph in r["training"]["phases"]:
            accs += [f"{e['val_acc']*100:.1f}%" for e in ph["epochs"]]
        lines.append(f"- **{r['backbone']}**: " + " → ".join(accs))

    lines += ["", "## Schwächste Klassen (niedrigster F1)", ""]
    for r in runs:
        worst = _worst_classes(r, k=3)
        parts = [f"{cls} ({m['f1']:.3f})" for cls, m in worst]
        lines.append(f"- **{r['backbone']}**: " + ", ".join(parts))

    lines += ["", "## Testbild-Vorhersagen (unabhängige Fotos)", ""]
    for r in runs:
        ti = r["test_images"]
        if not ti.get("predictions"):
            continue
        lines += [f"### {r['backbone']}", "",
                  "| Bild | Wahrheit | Vorhersage | Konfidenz | ✓ |",
                  "|---|---|---|---|---|"]
        for p in ti["predictions"]:
            mark = "✅" if p["correct"] else ("❌" if p["correct"] is False else "—")
            lines.append(
                f"| {p['image']} | {p['true'] or '—'} | {p['pred']} "
                f"| {p['confidence']*100:.1f}% | {mark} |"
            )
        lines.append("")

    # Diagramme verlinken (sofern bench/plot.py gelaufen ist).
    plots = [("Accuracy vs. Trainingszeit", "accuracy_vs_time.png"),
             ("Val-Accuracy je Epoche", "val_acc_curves.png")]
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
            lines.append(f"- **{r['backbone']}**: ![{r['backbone']}]({rel})")

    lines.append("")
    return "\n".join(lines)


def main():
    runs = load_all()
    md = build_markdown(runs)
    os.makedirs("results", exist_ok=True)
    out = "results/BENCHMARK.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"{len(runs)} Lauf/Läufe gefunden -> {out}")


if __name__ == "__main__":
    main()
