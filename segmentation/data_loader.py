"""Patch sampling and dataset utilities for U-Net training.

This module builds per-epoch patch pools with configurable positive/random
ratios for train and validation modes.
"""

from dataclasses import dataclass, asdict
import csv, os, json, time
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Optional
import pandas as pd
import albumentations as A
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Sequence
from config3 import *


class PatchSegDataset(Dataset):
    """Dataset that caches sampled patches for the active epoch.

    Modes:
    - train: resampled each epoch, uses warmup/late positive ratio schedule.
    - val_fixed: sampled once and reused for stable early-stopping metrics.
    - val_random: resampled each epoch for trend monitoring.
    """

    def __init__(self, df, mode,           # "train" | "val_fixed" | "val_random"
                use_aug=True):
        """Initialize patch-sampling dataset metadata and sampling policy."""
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.warmup_epochs = WARM_UP_EPOCHS
        self.total_epochs = NUM_EPOCHS
        self.patches_per_plane = PATCHES_PER_PLANE_PER_EPOCH
        self.val_patches_per_plane = VAL_PATCHES_PER_PLANE
        self.patch_size = PATCH_SIZE
        self.use_aug = use_aug and (A is not None)
        self.base_seed = BASE_SEED
        self.pos_ratio_warm = WARMUP_POS_RATIO
        self.pos_ratio_late = LATER_POS_RATIO
        self.val_fixed_ratio = VAL_FIXED_RATIO

        self.aug = default_augmentations() if (self.use_aug and self.mode=="train") else None

        self._epoch = -1
        self._patches_img = None
        self._patches_msk = None
        self._built_fixed = False  # for val_fixed

    def _epoch_rng(self, epoch: int):
        """Create deterministic epoch-specific RNG from base seed."""
        return np.random.default_rng(self.base_seed + epoch)

    def set_epoch(self, epoch: int):
        """Build or refresh patch pools according to dataset mode and epoch."""
        self._epoch = epoch
        if self.mode == "train":
            rng = self._epoch_rng(epoch)
            pos_ratio = (
                self.pos_ratio_warm
                if epoch < self.warmup_epochs
                else self.pos_ratio_late
            )
            self._build_patch_pool(
                pos_ratio=pos_ratio,
                rng=rng,
                do_aug=True,
                num_patches=self.patches_per_plane,
            )

        elif self.mode == "val_random":
            # Resample each epoch with a separate seed range from training.
            rng = self._epoch_rng(10_000 + epoch)
            self._build_patch_pool(
                pos_ratio=self.val_fixed_ratio,
                rng=rng,
                do_aug=False,
                num_patches=self.val_patches_per_plane,
            )

        elif self.mode == "val_fixed":
            # Build once and reuse for stable comparisons across epochs.
            if not self._built_fixed:
                rng = np.random.default_rng(self.base_seed + 777)
                self._build_patch_pool(
                    pos_ratio=self.val_fixed_ratio,
                    rng=rng,
                    do_aug=False,
                    num_patches=self.val_patches_per_plane,
                )
                self._built_fixed = True

    def _build_patch_pool(self, pos_ratio: float, rng, do_aug: bool, num_patches):
        """Sample patches for each plane and concatenate into cached arrays."""
        imgs, msks = [], []
        for _, row in self.df.iterrows():
            img = read_gray01(row.image_path)
            msk = read_mask01(row.mask_path)
            pi, pm = make_plane_patches_for_epoch(
                img, msk,
                pos_ratio=pos_ratio,
                rng=rng, num_patches=num_patches,
                aug=(self.aug if do_aug else None)         # <— IMPORTANT: pass epoch-tied RNG
            )
            imgs.append(pi); msks.append(pm)
        self._patches_img = np.concatenate(imgs, 0).astype(np.float32)
        self._patches_msk = np.concatenate(msks, 0).astype(np.float32)

    def __len__(self):
        """Return number of cached patches for current epoch pool."""
        assert self._patches_img is not None, \
            "Call set_epoch(epoch) at least once to build the patch pool."
        return self._patches_img.shape[0]

    def __getitem__(self, idx):
        """Return one patch pair as (1, H, W) tensors for image and mask."""
        im = self._patches_img[idx]
        mk = self._patches_msk[idx]
        return torch.from_numpy(im[None,...]), torch.from_numpy(mk[None,...])


def read_gray01(path: str, how="Normalised") -> np.ndarray:
    """Load grayscale image and optionally normalize to [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"Image not found: {path}"
    if how != "Normalised":
        return img
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn + 1e-8)
    return img

def read_mask01(path: str) -> np.ndarray:
    """Load segmentation mask as binary uint8 (0/1)."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert m is not None, f"Mask not found: {path}"
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = (m > 0).astype(np.uint8)
    return m


def make_candidate_map(mask01: np.ndarray) -> np.ndarray:
    """Dilate positives to define candidate centers for positive patch sampling."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * DILATE_RADIUS + 1, 2 * DILATE_RADIUS + 1),
    )
    cand = cv2.dilate(mask01, kernel, iterations=1)
    cand = (cand > 0).astype(np.uint8)
    return cand


def top_left_range_including_pixel(yc, xc, H, W):
    """Valid top-left patch-coordinate range that still contains pixel (yc, xc)."""
    y0_min = max(0, yc - (PATCH_SIZE - 1))
    y0_max = min(yc, H - PATCH_SIZE)
    x0_min = max(0, xc - (PATCH_SIZE - 1))
    x0_max = min(xc, W - PATCH_SIZE)
    return y0_min, y0_max, x0_min, x0_max


def sample_random_patch(
    img01: np.ndarray,
    mask01: np.ndarray,
    rng: np.random.Generator,
    max_retries: int = 10,
    forbid_coords: Optional[set] = None,
):
    """Sample a mostly-background patch, with retries for low-positive fraction."""
    H, W = img01.shape
    last = None
    for _ in range(max_retries):
        # NOTE: keeps original behavior by using numpy global RNG here.
        y0 = np.random.randint(0, H - PATCH_SIZE + 1)
        x0 = np.random.randint(0, W - PATCH_SIZE + 1)
        if forbid_coords is not None and (y0, x0) in forbid_coords:
            continue
        im = img01[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        mk = mask01[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        pos_frac = mk.mean()
        last = (im, mk, (y0, x0))
        if pos_frac <= RAND_MAX_FRAC:
            return im, mk, (y0, x0)
    # if we couldn't find a “pure enough” background, return the last
    return last


def sample_positive_patch(
    img01: np.ndarray,
    mask01: np.ndarray,
    cand: np.ndarray,
    rng: np.random.Generator,
    max_retries: int = 10,
):
    """Sample a patch near positive regions, ensuring minimum positive fraction."""

    H, W = img01.shape
    ys, xs = np.where(cand == 1)
    if len(ys) == 0:
        # Fallback when candidate map is empty.
        return sample_random_patch(img01, mask01, rng)

    for _ in range(max_retries):
        i = rng.integers(0, len(ys))
        yc, xc = int(ys[i]), int(xs[i])
        y0_min, y0_max, x0_min, x0_max = top_left_range_including_pixel(yc, xc, H, W)
        if y0_max < y0_min or x0_max < x0_min:
            continue
        y0 = rng.integers(y0_min, y0_max + 1)
        x0 = rng.integers(x0_min, x0_max + 1)

        im = img01[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        mk = mask01[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
        pos_frac = mk.mean()

        if pos_frac >= POS_MIN_FRAC:
            return im, mk, (y0, x0)

    # if retries fail, return last or fallback
    return im, mk, (y0, x0)
    
def make_plane_patches_for_epoch(
    img01,
    mask01,
    pos_ratio,
    rng: np.random.Generator,
    num_patches: int,
    aug=None,
):
    """Build one plane's patch set with a positive/random sampling mix."""
    cand = make_candidate_map(mask01)
    num_pos = int(round(num_patches * pos_ratio))
    num_rand = num_patches - num_pos

    patches_img, patches_msk = [], []
    used_pos_coords = set()

    # Positive-biased samples target likely cell regions.
    for _ in range(num_pos):
        im, mk, (y0, x0) = sample_positive_patch(img01, mask01, cand, rng=rng)
        used_pos_coords.add((y0, x0))
        if aug is not None:
            out = aug(image=im, mask=mk)
            im, mk = out["image"], out["mask"]
        patches_img.append(im)
        patches_msk.append(mk)

    # Random samples add harder background and context diversity.
    for _ in range(num_rand):
        im, mk, (y0, x0) = sample_random_patch(
            img01,
            mask01,
            rng=rng,
            forbid_coords=used_pos_coords,
        )
        if aug is not None:
            out = aug(image=im, mask=mk)
            im, mk = out["image"], out["mask"]
        patches_img.append(im)
        patches_msk.append(mk)

    patches_img = np.stack(patches_img, 0).astype(np.float32)  # [K, H, W]
    patches_msk = np.stack(patches_msk, 0).astype(np.float32)  # [K, H, W]
    return patches_img, patches_msk


def default_augmentations():
    """Return default Albumentations pipeline for patch training."""
    if A is None:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.10, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
    ])
