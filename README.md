# 🌱 Pflanzen- & Krankheitserkennung (Transfer Learning)

Projekt für das Modul **Maschinelles Sehen**. Ein vortrainiertes CNN (ImageNet)
wird per **Transfer Learning** auf den [PlantVillage]-Datensatz angepasst, um aus
einem Blattfoto die **Pflanzenart** und eine eventuelle **Krankheit** zu erkennen.

## Ansatz

- **Ein kombiniertes Klassifikationsmodell**: jede Klasse kodiert Pflanze *und*
  Zustand, z. B. `Tomato___Late_blight` oder `Apple___healthy` (38 Klassen,
  14 Pflanzenarten). Für die Ausgabe wird das Label am `___` in **Pflanze** und
  **Krankheit** zerlegt.
- **Backbone**: ResNet18 (Standard) oder EfficientNet-B0 (genauer), aus
  `torchvision`, vortrainiert auf ImageNet.
- **Zwei-Phasen-Training**:
  1. *Feature Extraction* – Backbone eingefroren, nur der neue Kopf lernt.
  2. *Fine-Tuning* – gesamtes Netz mit kleiner Lernrate nachtrainiert.

## Projektstruktur

```
src/data.py       Transforms + DataLoader (ImageFolder), Label-Split
src/model.py      build_model(): Backbone + neuer Kopf, Checkpoint speichern/laden
src/train.py      Zwei-Phasen-Trainingsloop, speichert bestes Modell
src/evaluate.py   Accuracy, Classification-Report, Confusion Matrix
src/predict.py    Einzelbild-Vorhersage -> (Pflanze, Krankheit, Konfidenz)
app/app.py        Gradio-Demo (lokal, CPU)
notebooks/        Colab-Notebook (Haupt-Deliverable, mit GPU trainieren)
```

## Schnellstart

### 1. Training (auf Google Colab mit GPU empfohlen)

Der lokale Rechner ist CPU-only → Training dort sehr langsam. Öffne
`notebooks/plant_disease_train.ipynb` in Colab (Runtime → GPU) und führe die
Zellen aus. Datensatz wird per Kaggle-API geladen (`kaggle.json` benötigt).

Alternativ als Skript:

```bash
pip install -r requirements.txt
python -m src.train --data-root data/plantvillage \
    --backbone resnet18 --epochs-head 3 --epochs-finetune 5
```

### 2. Inferenz auf einem Foto

```bash
python -m src.predict --model models/best_model.pth --image mein_foto.jpg
```

### 3. Interaktive Demo

```bash
python app/app.py    # öffnet eine lokale Weboberfläche
```

## Datensatz

[PlantVillage] – ~54 000 Blattbilder, 38 Klassen. Über Kaggle verfügbar
(z. B. *"New Plant Diseases Dataset"* mit fertigem `train/`–`valid/`-Split).

> **⚠️ Einschränkung:** PlantVillage besteht aus **Laboraufnahmen** (einzelnes
> Blatt, gleichmäßiger Hintergrund). Das Modell erreicht darauf sehr hohe
> Accuracy, generalisiert aber nur **eingeschränkt auf echte Feldfotos**. Für
> eine ehrliche Bewertung lohnt ein Test mit eigenen Handyfotos oder dem
> [PlantDoc]-Datensatz (Feldaufnahmen).

## Training auf AMD-GPU (Radeon RX 5700 XT)

`src/train.py` erkennt die Hardware automatisch (`get_device()`): CUDA / ROCm /
DirectML / MPS / CPU. Für die 5700 XT gibt es zwei Wege:

### Variante A – Windows + DirectML (empfohlen, am einfachsten)

```bash
git clone <DEINE_REPO_URL> && cd MS_Projekt
python -m venv .venv && .venv\Scripts\activate
pip install torch-directml            # bringt passendes torch mit
pip install -r requirements.txt       # torch/torchvision werden nicht ueberschrieben
python -m src.train --data-root data/<datensatz> --backbone resnet18
```

`torch_directml` wird automatisch genutzt, sobald installiert. Funktioniert mit
jeder DirectX-12-GPU – ideal für Navi-Karten wie die 5700 XT unter Windows.

### Variante B – Linux + ROCm

Die 5700 XT (gfx1010) wird von aktuellem ROCm offiziell **nicht** unterstützt,
läuft aber oft mit einem Override. PyTorch (ROCm-Build) meldet sich dann als
`cuda`, der Code braucht keine Änderung:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # noetiger Workaround fuer Navi 10
python -m src.train --data-root data/<datensatz> --backbone resnet18
```

> Wenn ROCm zickt: einfach Variante A (DirectML) nehmen.

## Setup-Hinweise

- **GPU**: Colab/Kaggle bieten kostenlose NVIDIA-GPUs (kein AMD-Setup nötig).
  Der Mac (Intel, integrierte Grafik) hat weder CUDA noch MPS → nur Inferenz/Demo.
- **Kaggle-Token**: Konto → *Account* → *Create New API Token* lädt `kaggle.json`.
  In Colab hochladen, nach `~/.kaggle/kaggle.json` kopieren, `chmod 600`.

## Mögliche Erweiterungen

- ResNet18 vs. EfficientNet-B0 vergleichen (Accuracy / Parameter / Zeit).
- **Grad-CAM**: visualisieren, worauf das Netz achtet.
- Multi-Task-Variante mit getrennten Köpfen für Pflanze und Krankheit.

[PlantVillage]: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
[PlantDoc]: https://github.com/pratikkayal/PlantDoc-Dataset
