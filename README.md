# miniDepth

A three-part toolkit for calcium-imaging cell analysis:

| Package | Purpose |
|---|---|
| [`gui/`](gui/README.md) | Interactive Napari viewer — explore 3D column labels, match MINIAN footprints, export stats |
| [`segmentation/`](segmentation/README.md) | U-Net pipeline — preprocess raw TIFF stacks, train, run inference, and evaluate |
| [`global_mask_pipeline/`](global_mask_pipeline/README.md) | Cross-plane post-processing to generate globally consistent `GLOBAL_PLANE_MASKS` |
| [`U-Net/`](U-Net/README.md) | Legacy folder with the original script-based U-Net pipeline |

---

## Repository Layout

```
TOOL/
├── README.md               ← this file
├── requirements.txt        ← all Python dependencies
│
├── gui/                    ← interactive analysis GUI
│   ├── __init__.py
│   ├── main.py             ← entry point  (python -m gui <data_dir>)
│   ├── core.py             ← volume loading, overlap math, MINIAN I/O
│   ├── viewer.py           ← Napari controller
│   ├── widgets.py          ← Qt dock widgets
│   ├── stats.py            ← batch CSV statistics
│   └── README.md           ← GUI-specific docs
│
├── segmentation/           ← U-Net training & inference
│   ├── __init__.py
│   ├── config.py           ← hyperparameters
│   ├── preprocess.py       ← TIFF stack preprocessing
│   ├── data_loader.py      ← patch-sampling Dataset
│   ├── model.py            ← U-Net architecture + losses
│   ├── train.py            ← training entry point
│   ├── predict.py          ← tiled inference + evaluation
│   ├── region_growing.py   ← classical alternative segmentation
│   └── README.md           ← segmentation-specific docs
│
├── global_mask_pipeline/   ← cross-plane chain + global plane-mask generation
│   ├── __init__.py
│   ├── save_global_plane_masks.py
│   └── README.md
│
├── U-Net/                  ← legacy script layout (kept for compatibility)
│   └── README.md
│
├── data/                   ← raw / preprocessed data (not committed)
└── outputs/                ← model checkpoints, CSVs, overlays (not committed)
```

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU training, install the appropriate CUDA-enabled PyTorch build from https://pytorch.org first.

---

## Quick Start

### Launch the GUI viewer

```bash
python -m gui "G:\path\to\SG006_3D_D3"
```

See [gui/README.md](gui/README.md) for the full interactive workflow.

### Run segmentation training

```bash
python -m segmentation.train
```

See [segmentation/README.md](segmentation/README.md) for preprocessing, training, and inference steps.

### Build global plane masks

```bash
python -m global_mask_pipeline.save_global_plane_masks
```

See [global_mask_pipeline/README.md](global_mask_pipeline/README.md) for required input/output structure.

---

## Data Conventions

- Label volumes are `(z, y, x)` NumPy arrays; background = label `0`.
- Global plane masks: `GLOBAL_PLANE_MASKS/plane_001_global.npy`, `plane_002_global.npy`, …
- Preprocessed TIFFs: `PRE_PROCESSED_TIFF/<mouse_id>_P1.tif`, `_P2.tif`, …
- MINIAN data: a folder with Zarr subdirectories containing an `A` array with a `unit_id` coordinate.
