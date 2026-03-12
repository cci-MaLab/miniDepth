"""Training entry point for U-Net patch segmentation.

This script:
- loads split metadata,
- builds train/validation datasets and loaders,
- trains with mixed precision,
- logs metrics and checkpoints per run.
"""

from . import config
from .data_loader import PatchSegDataset
from . import data_loader
from .model import *
from .model import RunLogger, UNet
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Sequence
import os
from .config import *
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")


def seed_worker(worker_id):
    """Set deterministic numpy/random seeds for each dataloader worker."""
    # Tie torch/np/random to the worker-specific PyTorch seed.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    print("Reading Data...")
    df_all = pd.read_csv("data_pairs_dz10.csv")  # columns: image_path, mask_path, plane_index, split
    # Split DataFrames
    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)   # 20 planes
    df_val   = df_all[df_all["split"] == "val"].reset_index(drop=True)     # 3 planes

    # Instantiate datasets
    train_ds     = PatchSegDataset(df_train, mode="train")
    val_fixed_ds = PatchSegDataset(df_val,   mode="val_fixed")   # stable early-stop set
    val_rand_ds  = PatchSegDataset(df_val,   mode="val_random")   # resampled each epoch (monitor only)
    print("Data Datasets Created....")

    # Build the initial pools
    train_ds.set_epoch(0)       # warmup ratio
    val_fixed_ds.set_epoch(0)   # builds once and caches (deterministic)
    val_rand_ds.set_epoch(0)    # builds epoch-0 random set

    #Now create DataLoaders
    train_loader     = DataLoader(train_ds,     batch_size=12, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
    val_fixed_loader = DataLoader(val_fixed_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker)
    val_rand_loader  = DataLoader(val_rand_ds,  batch_size=16, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker)
    print("Data Loaders Created....")

    model = UNet(in_ch=1, out_ch=1, base=32, norm="bn")
    print("Model Created....")

    # ----- logging & ckpts -----
    run = RunLogger(run_dir="runs_unet")
    best_path = os.path.join(run.dir, "best.ckpt")
    last_path = os.path.join(run.dir, "last.ckpt")
    print("RunLogger Created....")

    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    max_grad_norm = 1.0
    best_loss = 1000000000
    patience = 0

    print("Training...")
    for ep in range(NUM_EPOCHS):

        # ========== TRAIN ==========
        model.train()
        train_ds.set_epoch(ep)   # resample new train patches (and switch out of warmup when time)
        running_loss, seen = 0.0, 0

        for imgs, masks in train_loader:
            imgs  = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = calculate_loss(logits, masks)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            seen         += imgs.size(0)

        train_loss = running_loss / max(1, seen)
        if sched is not None:
            sched.step()

        # ========== VALIDATION (fixed) ==========
        model.eval()
        valf_loss, valf_dice, valf_iou, valf_ap = eval_loader(model, val_fixed_loader)

        # ========== VALIDATION (random monitor; trend only) ==========
        val_rand_ds.set_epoch(ep)  # resample a fresh random set each epoch
        valr_loss, valr_dice, valr_iou, valr_ap = eval_loader(model, val_rand_loader)

        # ----- print + log -----
        curr_lr = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else opt.param_groups[0]["lr"]
        print(
            f"ep{ep}: "
            f"train_loss={train_loss:.4f}  "
            f"| valF loss={valf_loss:.4f} dice={valf_dice:.4f} iou={valf_iou:.4f} ap={valf_ap:.4f}  "
            f"| valR loss={valr_loss:.4f} dice={valr_dice:.4f} iou={valr_iou:.4f} ap={valr_ap:.4f}  "
            f"| lr={curr_lr:.2e}"
        )

        run.add(EpochLog(
            epoch=ep, train_loss=train_loss,
            valf_loss=valf_loss, valf_dice=valf_dice, valf_iou=valf_iou, valf_ap=valf_ap,
            valr_loss=valr_loss, valr_dice=valr_dice, valr_iou=valr_iou, valr_ap=valr_ap,
            lr=curr_lr
        ))

        # ----- checkpoints (best by val_fixed loss) -----
        if valf_loss < best_loss:
            best_loss = valf_loss
            patience  = 0
            save_ckpt(best_path, model, opt, sched, ep, best_loss)
        else:
            patience += 1

        # always save last
        save_ckpt(last_path, model, opt, sched, ep, best_loss)

        # ----- early stopping -----
        if patience >= PATIENCE_LIMIT:
            print("Early stopping.")
            break

    # finalize: plots + paths
    run.plot()
    print("Artifacts saved in:", run.dir)
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)
