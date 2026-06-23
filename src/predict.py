"""Inferenz auf einem Einzelbild + Aufteilung in Pflanze und Krankheit.

CLI-Beispiel:
    python -m src.predict --model models/best_model.pth --image foto.jpg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms, split_label
from .model import load_checkpoint


@dataclass
class Prediction:
    plant: str          # z.B. "Tomato"
    condition: str      # z.B. "Late blight" oder "healthy"
    label: str          # rohes Klassenlabel, z.B. "Tomato___Late_blight"
    confidence: float   # Wahrscheinlichkeit der Top-Klasse (0..1)
    is_healthy: bool

    def __str__(self) -> str:
        status = "gesund" if self.is_healthy else f"Krankheit: {self.condition}"
        return f"Pflanze: {self.plant} | {status} ({self.confidence * 100:.1f}%)"


def predict_image(image_path_or_pil, model, class_names, device=None, top_k=3):
    """Sagt Pflanze + Zustand fuer ein einzelnes Bild vorher.

    Parameters
    ----------
    image_path_or_pil : Pfad (str) oder bereits geladenes PIL.Image.
    model, class_names : aus load_checkpoint().
    top_k : wie viele Top-Vorhersagen zusaetzlich zurueckgegeben werden.

    Returns
    -------
    best : Prediction
    topk : Liste von (label, wahrscheinlichkeit), absteigend sortiert.
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(image_path_or_pil, (str, bytes)):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")

    transform = build_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0)

    k = min(top_k, len(class_names))
    top_probs, top_idx = probs.topk(k)
    topk = [(class_names[i], float(p)) for p, i in zip(top_probs, top_idx)]

    best_label, best_conf = topk[0]
    plant, condition = split_label(best_label)
    is_healthy = condition.lower() in ("healthy", "gesund")

    best = Prediction(
        plant=plant,
        condition=condition,
        label=best_label,
        confidence=best_conf,
        is_healthy=is_healthy,
    )
    return best, topk


def main():
    parser = argparse.ArgumentParser(description="Einzelbild-Vorhersage")
    parser.add_argument("--model", default="models/best_model.pth")
    parser.add_argument("--image", required=True, help="Pfad zum Foto")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_checkpoint(args.model, device=device)
    best, topk = predict_image(
        args.image, model, class_names, device=device, top_k=args.top_k
    )

    print(best)
    print("\nTop-Vorhersagen:")
    for label, prob in topk:
        print(f"  {label:40s} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
