# segmentation — U-Net Cell Segmentation Pipeline

End-to-end pipeline for segmenting cells in calcium-imaging TIFF planes: preprocessing → patch-based training → tiled inference → evaluation. Also includes a classical region-growing alternative.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters and loss configuration |
| `preprocess.py` | Full TIFF stack preprocessing (denoising, motion correction, projection) |
| `data_loader.py` | Patch-sampling `Dataset` for training and validation |
| `model.py` | U-Net architecture, loss functions, metrics, checkpointing, run logging |
| `train.py` | Training loop with mixed precision and early stopping |
| `predict.py` | Tiled inference, thresholding, multi-threshold evaluation |
| `region_growing.py` | Classical alternative: local maxima → radial edges → BFS region grow |

---

## Pipeline Overview

```
Raw TIFF stacks
      │
      ▼
preprocess.py          glow removal → denoising → motion correction → projection
      │
      ▼
data_pairs_dz10.csv    image_path | mask_path | plane_index | split
      │
      ├──▶ data_loader.py   patch sampling (positive + random, with augmentation)
      │
      ├──▶ train.py         mixed-precision training, best/last checkpoints
      │         └──▶ model.py  (UNet, losses, RunLogger)
      │
      └──▶ predict.py       tiled overlap-add inference, threshold sweep, overlays
```

---

## Step 1 — Preprocessing

Edit the paths in `preprocess.py` then run:

```bash
python -m segmentation.preprocess
```

Per-stack pipeline:
1. Load middle N frames from raw TIFF stack.
2. Morphological opening (glow removal).
3. Median + Gaussian + temporal-median + speckle-opening denoising.
4. Rigid subpixel phase-correlation motion correction (optional 2-pass).
5. Temporal mean projection.
6. White top-hat background subtraction.
7. Save QC intermediates and final output TIFF.

---

## Step 2 — Prepare the Split CSV

Create a CSV file named `data_pairs_dz10.csv` with columns:

| Column | Description |
|---|---|
| `image_path` | Absolute path to preprocessed TIFF plane |
| `mask_path` | Absolute path to binary ground-truth mask TIFF |
| `plane_index` | Integer plane index (used for train/val distance metrics) |
| `split` | `"train"` or `"val"` |

---

## Step 3 — Configure Hyperparameters

Edit `config.py` to set:

| Parameter | Default | Description |
|---|---|---|
| `PATCH_SIZE` | `256` | Square patch size (pixels) |
| `NUM_EPOCHS` | `35` | Maximum training epochs |
| `WARM_UP_EPOCHS` | `10` | Epochs using high positive-patch ratio |
| `LOSS` | `"Tversky"` | `"Dice+BCE"`, `"Tversky"`, `"Focal-Tversky"`, `"IOU"` |
| `ALPHA` / `BETA` | `0.3` / `0.7` | Tversky FP / FN weights |
| `PATIENCE_LIMIT` | `5` | Early stopping patience |
| `DEVICE` | `"cuda"` | `"cuda"` or `"cpu"` |

---

## Step 4 — Train

```bash
python -m segmentation.train
```

Outputs are written to `runs_unet/<run_id>/`:

```
runs_unet/<run_id>/
├── best.ckpt       ← best val_fixed loss
├── last.ckpt       ← last completed epoch
├── metrics.csv
├── metrics.json
├── loss.png
├── dice.png
├── iou.png
└── ap.png
```

---

## Step 5 — Inference & Evaluation

```bash
python -m segmentation.predict
```

- Tiled overlap-and-average prediction (tile = 256 px, overlap = 64 px).
- Threshold modes: `"otsu"`, `"global"`, `"sweep"` (best-Dice over a grid).
- Saves colour-coded TP/FP/FN overlay PNGs and binary mask TIFFs.

---

## Step 6 — Global Plane Mask Pipeline

If you generate per-cell labeled masks and want consistent global IDs across planes, use:

```bash
python -m global_mask_pipeline.save_global_plane_masks
```

This workflow is implemented in the top-level folder [../global_mask_pipeline/README.md](../global_mask_pipeline/README.md) and runs:
1. Cell mask extraction and centroid export.
2. Cross-plane chain construction.
3. Global ID assignment per chain.
4. Export of `GLOBAL_PLANE_MASKS/plane_XXX_global.npy` files.

---

## Classical Alternative: Region Growing

`region_growing.py` provides a non-deep-learning segmentation path:

1. Detect cell seeds via `skimage.feature.peak_local_max`.
2. Cast radial rays (Bresenham lines) to find edge pixels.
3. BFS region grow constrained by distance, intensity, and gradient monotonicity.

```bash
python -m segmentation.region_growing
```

---

## Model Architecture (U-Net)

```
Input (1 × H × W)
  enc1 (base)   ──────────────────────────────────┐
  enc2 (2×base)  ─────────────────────────────┐   │
  enc3 (4×base)  ─────────────────────────┐   │   │  skip connections
  enc4 (8×base)  ─────────────────────┐   │   │   │
  bottleneck (16×base, optional dropout)   │   │   │
  dec4 ────────────────────────────────────┘   │   │
  dec3 ────────────────────────────────────────┘   │
  dec2 ────────────────────────────────────────────┘
  dec1 (base)
  1×1 Conv → raw logits  (sigmoid applied externally)
```

Default: `base = 32`, batch norm, no dropout. Override via `UNet(in_ch=1, out_ch=1, base=32, norm="bn")`.
