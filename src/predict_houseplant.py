"""Kombinierte Zimmerpflanzen-Inferenz: Art -> Zustand.

Schickt ein Foto nacheinander durch zwei getrennte Modelle:
1. Art-Modell    (``models/houseplant_species.pth``) -> welche Zimmerpflanze?
2. Zustand-Modell (``models/houseplant_health.pth``)  -> gesund oder welk?

Bewusst zwei Modelle statt eines kombinierten Klassifikators, weil
Krankheitsdaten fuer Zimmerpflanzen kaum existieren (siehe Projektplan).

CLI-Beispiel:
    python -m src.predict_houseplant --image zimmerpflanze_kevin_1.jpeg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms
from .model import load_checkpoint

DEFAULT_SPECIES_MODEL = "models/houseplant_species.pth"
DEFAULT_HEALTH_MODEL = "models/houseplant_health.pth"

# Labels, die als "gesund" gelten (Zustand-Modell ist binaer gesund/welk).
HEALTHY_LABELS = {"healthy", "gesund"}


@dataclass
class HouseplantPrediction:
    species: str            # z.B. "Snake plant (Sanseviera)"
    species_confidence: float
    condition: str          # rohes Zustand-Label, z.B. "healthy" / "wilted"
    condition_confidence: float
    is_healthy: bool

    def __str__(self) -> str:
        status = "gesund" if self.is_healthy else "welk / kraenkelnd"
        return (
            f"Art: {self.species} ({self.species_confidence * 100:.1f}%) | "
            f"Zustand: {status} ({self.condition_confidence * 100:.1f}%)"
        )


def _classify(image, model, class_names, device, top_k=3):
    """Softmax-Vorhersage fuer ein bereits geladenes PIL-Bild.

    Returns
    -------
    topk : Liste von (label, wahrscheinlichkeit), absteigend sortiert.
    """
    transform = build_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0)

    k = min(top_k, len(class_names))
    top_probs, top_idx = probs.topk(k)
    return [(class_names[i], float(p)) for p, i in zip(top_probs, top_idx)]


def predict_houseplant(
    image_path_or_pil,
    species_model,
    species_classes,
    health_model,
    health_classes,
    device=None,
    top_k=3,
):
    """Fuehrt ein Bild durch Art- und Zustand-Modell.

    Returns
    -------
    best : HouseplantPrediction
    species_topk : Liste von (art, wahrscheinlichkeit)
    health_topk  : Liste von (zustand, wahrscheinlichkeit)
    """
    if device is None:
        device = next(species_model.parameters()).device

    if isinstance(image_path_or_pil, (str, bytes)):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")

    species_topk = _classify(image, species_model, species_classes, device, top_k)
    health_topk = _classify(image, health_model, health_classes, device, top_k)

    species, species_conf = species_topk[0]
    condition, condition_conf = health_topk[0]
    is_healthy = condition.lower() in HEALTHY_LABELS

    best = HouseplantPrediction(
        species=species,
        species_confidence=species_conf,
        condition=condition,
        condition_confidence=condition_conf,
        is_healthy=is_healthy,
    )
    return best, species_topk, health_topk


def main():
    parser = argparse.ArgumentParser(
        description="Kombinierte Zimmerpflanzen-Inferenz (Art -> Zustand)"
    )
    parser.add_argument("--image", required=True, help="Pfad zum Foto")
    parser.add_argument("--species-model", default=DEFAULT_SPECIES_MODEL)
    parser.add_argument("--health-model", default=DEFAULT_HEALTH_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    # Inferenz laeuft auf CPU (DirectML nur fuers Training genutzt).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    species_model, species_classes = load_checkpoint(args.species_model, device=device)
    health_model, health_classes = load_checkpoint(args.health_model, device=device)

    best, species_topk, health_topk = predict_houseplant(
        args.image,
        species_model,
        species_classes,
        health_model,
        health_classes,
        device=device,
        top_k=args.top_k,
    )

    print(best)
    print("\nTop-Arten:")
    for label, prob in species_topk:
        print(f"  {label:45s} {prob * 100:5.1f}%")
    print("\nZustand:")
    for label, prob in health_topk:
        print(f"  {label:45s} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
