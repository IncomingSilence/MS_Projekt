"""Modellaufbau fuer Transfer Learning.

Wir laden ein vortrainiertes Backbone (ImageNet) und ersetzen den
Klassifikationskopf durch einen neuen Linear-Layer mit ``num_classes`` Ausgaengen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet18", "efficientnet_b0")


def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Erzeugt ein vortrainiertes Netz mit ausgetauschtem Klassifikationskopf.

    Parameters
    ----------
    num_classes : Anzahl der Zielklassen (z.B. 38 bei PlantVillage).
    backbone : "resnet18" (schnell, einsteigerfreundlich) oder
        "efficientnet_b0" (etwas genauer).
    freeze_backbone : Wenn True, werden alle Schichten ausser dem neuen Kopf
        eingefroren (Feature-Extraction-Phase).
    pretrained : ImageNet-Gewichte laden (sonst zufaellige Initialisierung).
    """
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unbekanntes Backbone '{backbone}'. Erlaubt: {SUPPORTED_BACKBONES}"
        )

    return model


def unfreeze_all(model: nn.Module) -> None:
    """Alle Parameter wieder trainierbar machen (fuer die Fine-Tuning-Phase)."""
    for param in model.parameters():
        param.requires_grad = True


def trainable_parameters(model: nn.Module):
    """Iterator ueber die aktuell trainierbaren Parameter (fuer den Optimizer)."""
    return (p for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str, model: nn.Module, class_names, backbone: str) -> None:
    """Speichert Gewichte + Metadaten, damit predict.py alles rekonstruieren kann."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "backbone": backbone,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu"):
    """Laedt einen Checkpoint und baut das passende Modell wieder auf.

    Returns
    -------
    model (eval-Modus), class_names
    """
    ckpt = torch.load(path, map_location=device)
    class_names = ckpt["class_names"]
    backbone = ckpt.get("backbone", "resnet18")
    model = build_model(
        num_classes=len(class_names),
        backbone=backbone,
        freeze_backbone=False,
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
