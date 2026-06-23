"""Lokale Gradio-Demo: Foto hochladen -> Pflanze + Krankheit.

Start:
    python app/app.py            # nutzt models/best_model.pth
    MODEL_PATH=models/x.pth python app/app.py

Laeuft auf der CPU voellig ausreichend schnell (nur Inferenz).
"""

from __future__ import annotations

import os
import sys

# Projekt-Root in den Pfad, damit "from src..." auch beim direkten Aufruf klappt.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr  # noqa: E402

from src.model import load_checkpoint  # noqa: E402
from src.predict import predict_image  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")

_model = None
_class_names = None


def _ensure_model():
    global _model, _class_names
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modell nicht gefunden: {MODEL_PATH}. "
                "Zuerst trainieren (siehe Notebook) oder MODEL_PATH setzen."
            )
        _model, _class_names = load_checkpoint(MODEL_PATH, device="cpu")
    return _model, _class_names


def classify(image):
    if image is None:
        return "Bitte ein Bild hochladen.", {}
    model, class_names = _ensure_model()
    best, topk = predict_image(image, model, class_names, device="cpu", top_k=3)

    status = "gesund ✅" if best.is_healthy else f"Krankheit: {best.condition} ⚠️"
    summary = (
        f"## {best.plant}\n"
        f"**Zustand:** {status}\n\n"
        f"**Konfidenz:** {best.confidence * 100:.1f}%"
    )
    # gr.Label erwartet {Klasse: Wahrscheinlichkeit}.
    topk_dict = {label: prob for label, prob in topk}
    return summary, topk_dict


def build_demo():
    with gr.Blocks(title="Pflanzen- & Krankheitserkennung") as demo:
        gr.Markdown(
            "# 🌱 Pflanzen- & Krankheitserkennung\n"
            "Lade ein Blattfoto hoch. Das Modell erkennt die Pflanzenart und ob "
            "eine Krankheit vorliegt.\n\n"
            "> Hinweis: Trainiert auf PlantVillage (Laboraufnahmen). Auf echten "
            "Feldfotos kann die Genauigkeit geringer sein."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Blattfoto")
            with gr.Column():
                out_summary = gr.Markdown()
                out_topk = gr.Label(num_top_classes=3, label="Top-3")
        btn = gr.Button("Analysieren", variant="primary")
        btn.click(classify, inputs=inp, outputs=[out_summary, out_topk])
        inp.change(classify, inputs=inp, outputs=[out_summary, out_topk])
    return demo


if __name__ == "__main__":
    build_demo().launch()
