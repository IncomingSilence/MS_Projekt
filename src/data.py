"""Datenpipeline: Transforms und DataLoader fuer den PlantVillage-Datensatz.

Der Datensatz wird als ImageFolder erwartet, d.h. ein Ordner pro Klasse, wobei
der Ordnername das Label ist (z.B. ``Tomato___Late_blight``). Aus diesem Label
leiten wir spaeter Pflanze und Krankheit per ``"___"``-Split ab.

Zwei Layouts werden unterstuetzt:
1. Bereits gesplittet:  <root>/train/<klasse>/...  und  <root>/valid/<klasse>/...
2. Ein einzelner Ordner: <root>/<klasse>/...   -> wird hier zufaellig gesplittet.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

# ImageNet-Statistik: die vortrainierten Backbones erwarten genau diese Normalisierung.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def build_transforms(train: bool) -> transforms.Compose:
    """Transform-Pipeline. Beim Training mit Augmentierung, sonst deterministisch."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _has_split(root: str) -> bool:
    """True, wenn <root> bereits train/ und valid/ (oder val/) Unterordner hat."""
    has_train = os.path.isdir(os.path.join(root, "train"))
    has_valid = os.path.isdir(os.path.join(root, "valid")) or os.path.isdir(
        os.path.join(root, "val")
    )
    return has_train and has_valid


def _valid_dir(root: str) -> str:
    for name in ("valid", "val"):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Kein valid/ oder val/ Ordner unter {root}")


def _train_targets(train_ds) -> list[int]:
    """Liefert die Klassen-Labels aller Train-Samples (auch bei Subset)."""
    if isinstance(train_ds, torch.utils.data.Subset):
        base = train_ds.dataset
        return [base.targets[i] for i in train_ds.indices]
    return list(train_ds.targets)


def _build_weighted_sampler(train_ds) -> WeightedRandomSampler:
    """Sampler, der seltene Klassen haeufiger zieht (gegen Klassen-Ungleichgewicht).

    Sample-Gewicht = 1 / Haeufigkeit der eigenen Klasse, sodass im Erwartungswert
    jede Klasse gleich oft im Batch landet.
    """
    targets = _train_targets(train_ds)
    counts = Counter(targets)
    class_weight = {cls: 1.0 / n for cls, n in counts.items()}
    sample_weights = [class_weight[t] for t in targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    val_split: float = 0.2,
    seed: int = 42,
    weighted_sampler: bool = False,
):
    """Erzeugt Train-/Val-DataLoader und gibt zusaetzlich die Klassennamen zurueck.

    Mit ``weighted_sampler=True`` wird statt zufaelligem Shuffle ein
    WeightedRandomSampler verwendet, der seltene Klassen haeufiger zieht
    (sinnvoll bei stark unausgewogenen Datensaetzen wie den Zimmerpflanzen-Arten).

    Returns
    -------
    train_loader, val_loader, class_names
    """
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    pin_memory = torch.cuda.is_available()

    if _has_split(data_root):
        train_ds = datasets.ImageFolder(
            os.path.join(data_root, "train"), transform=build_transforms(train=True)
        )
        val_ds = datasets.ImageFolder(
            _valid_dir(data_root), transform=build_transforms(train=False)
        )
        class_names = train_ds.classes
    else:
        # Ein einzelner Ordner -> selbst splitten. Wir laden den Datensatz zweimal
        # (mit Train- bzw. Val-Transform) und teilen mit identischem Seed, damit
        # die Indizes konsistent bleiben.
        full_train = datasets.ImageFolder(
            data_root, transform=build_transforms(train=True)
        )
        full_val = datasets.ImageFolder(
            data_root, transform=build_transforms(train=False)
        )
        class_names = full_train.classes

        n_total = len(full_train)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(seed)
        train_idx, val_idx = random_split(
            range(n_total), [n_train, n_val], generator=generator
        )
        train_ds = torch.utils.data.Subset(full_train, list(train_idx))
        val_ds = torch.utils.data.Subset(full_val, list(val_idx))

    # Sampler und shuffle schliessen sich gegenseitig aus.
    sampler = _build_weighted_sampler(train_ds) if weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, class_names


def split_label(label: str) -> tuple[str, str]:
    """Zerlegt ein PlantVillage-Label in (Pflanze, Zustand/Krankheit).

    >>> split_label("Tomato___Late_blight")
    ('Tomato', 'Late blight')
    >>> split_label("Apple___healthy")
    ('Apple', 'healthy')
    """
    if "___" in label:
        plant, disease = label.split("___", 1)
    else:
        plant, disease = label, "unknown"
    # Unterstriche zu Leerzeichen fuer schoenere Anzeige.
    return plant.replace("_", " ").strip(), disease.replace("_", " ").strip()
