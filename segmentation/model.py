"""U-Net model, training losses, metrics, and run logging utilities.

This module groups:
- checkpoint save/load helpers,
- epoch-level logging and plotting,
- segmentation losses/metrics,
- U-Net building blocks and model definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dataclasses import dataclass, asdict
import csv, os, json, time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from config3 import *
import os
import cv2


def save_ckpt(path, model, optimizer, scheduler, epoch, best_metric):
    """Serialize model state and optimizer/scheduler state to checkpoint."""
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric
    }, path)

def load_ckpt(path, model, optimizer=None, scheduler=None, map_location="cuda"):
    """Load checkpoint into model and optionally optimizer/scheduler."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)


@dataclass
class EpochLog:
    """Container for one epoch of train/validation metrics and learning rate."""
    epoch: int
    train_loss: float
    valf_loss: float
    valf_dice: float
    valf_iou: float
    valf_ap: float
    valr_loss: float
    valr_dice: float
    valr_iou: float
    valr_ap: float
    lr: float

class RunLogger:
    """Store per-epoch metrics and export summary plots for a training run."""

    def __init__(self, run_dir="runs"):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(run_dir, ts)
        os.makedirs(self.dir, exist_ok=True)
        self.rows = []
        self.csv_path = os.path.join(self.dir, "metrics.csv")
        self.json_path = os.path.join(self.dir, "metrics.json")

    def add(self, row: EpochLog):
        """Append one epoch record and mirror history to CSV/JSON."""
        self.rows.append(asdict(row))
        # Append one row to CSV each call and write header once.
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.rows[-1].keys())
            if write_header:
                w.writeheader()
            w.writerow(self.rows[-1])

        # Keep a full JSON history for easy downstream parsing.
        with open(self.json_path, "w") as f:
            json.dump(self.rows, f, indent=2)

    def plot(self):
        """Generate and save loss/metric trend plots in the run directory."""
        df = pd.read_csv(self.csv_path)
        # Losses
        plt.figure(); df.plot(x="epoch", y=["train_loss","valf_loss","valr_loss"]); plt.title("Loss"); plt.savefig(os.path.join(self.dir,"loss.png")); plt.close()
        # Dice
        plt.figure(); df.plot(x="epoch", y=["valf_dice","valr_dice"]); plt.title("Dice"); plt.savefig(os.path.join(self.dir,"dice.png")); plt.close()
        # IoU
        plt.figure(); df.plot(x="epoch", y=["valf_iou","valr_iou"]); plt.title("IoU"); plt.savefig(os.path.join(self.dir,"iou.png")); plt.close()
        # AP
        plt.figure(); df.plot(x="epoch", y=["valf_ap","valr_ap"]); plt.title("PR-AUC (Average Precision)"); plt.savefig(os.path.join(self.dir,"ap.png")); plt.close()

def eval_loader(model, loader):
    """Return avg loss, Dice, IoU, AP over a loader."""
    model.eval()
    n, tot_loss, tot_dice, tot_iou, all_probs, all_gts = 0, 0.0, 0.0, 0.0, [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)

            if LOSS == "Dice+BCE":
                loss = bce_dice_loss(logits, masks)
            elif LOSS == "Tversky":
                loss = tversky_loss(logits, masks)
            elif LOSS == "Focal-Tversky":
                loss = focal_tversky_loss(logits, masks)
            elif LOSS == "IOU":
                loss = iou_loss(logits, masks)

            d, i = dice_iou_scores(logits, masks)
            tot_loss += loss.item() * imgs.size(0)
            tot_dice += d * imgs.size(0)
            tot_iou  += i * imgs.size(0)
            # collect for AP
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(imgs.size(0), -1)
            gts   = masks.detach().cpu().numpy().reshape(imgs.size(0), -1)
            all_probs.append(probs); all_gts.append(gts)
            n += imgs.size(0)
    avg_loss = tot_loss / max(1,n)
    avg_dice = tot_dice / max(1,n)
    avg_iou  = tot_iou  / max(1,n)
    probs = np.concatenate(all_probs, axis=0)
    gts   = np.concatenate(all_gts, axis=0)
    # compute AP per-sample then average (robust for class imbalance)
    aps = []
    for p, t in zip(probs, gts):
        if t.sum() == 0:  # degenerate: no positives -> define AP safely
            # treat as 1 - FP prominence; fallback to area under (1 - p) vs t is undefined
            # practical choice: skip or count AP=1.0 for empty GT with near-zero preds
            if p.mean() < 1e-3: aps.append(1.0)
            else: aps.append(0.0)
        else:
            aps.append(average_precision_score(t, p))
    avg_ap = float(np.mean(aps)) if len(aps) else 0.0
    return avg_loss, avg_dice, avg_iou, avg_ap


def calculate_loss(logits, target):
    """Dispatch to the configured segmentation loss."""
    if LOSS == "Dice+BCE":
        return bce_dice_loss(logits, target)
    elif LOSS == "Tversky":
        return tversky_loss(logits, target)
    elif LOSS == "Focal-Tversky":
        return focal_tversky_loss(logits, target)
    elif LOSS == "IOU":
        return iou_loss(logits, target)


def dice_loss(pred_sigmoid, target):
    """Compute soft Dice loss from probabilities and binary targets."""
    num = 2 * (pred_sigmoid * target).sum()
    den = pred_sigmoid.sum() + target.sum() + EPS
    return 1 - num / den


def bce_dice_loss(logits, target):
    """Balanced BCE + Dice loss on raw logits."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(torch.sigmoid(logits), target)
    return 0.5 * bce + 0.5 * dice


def tversky_loss(y_pred, y_true, alpha=ALPHA, beta=BETA):
    """Compute Tversky loss with configurable false-positive/negative weights."""
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()

    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    tversky = (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)
    return 1 - tversky.mean()


def focal_tversky_loss(y_pred, y_true):
    """Focal-Tversky variant to focus optimization on harder examples."""
    tv_loss = tversky_loss(y_pred, y_true, ALPHA, BETA)
    return tv_loss.pow(GAMMA)


def iou_loss(y_pred, y_true):
    """Compute soft IoU loss from logits and binary targets."""
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return 1 - (intersection + EPS) / (union + EPS)


def otsu_threshold(prob: np.ndarray) -> float:
    """Return Otsu threshold in [0, 1] for a probability map."""
    if isinstance(prob, torch.Tensor):
        prob = prob.detach().cpu().numpy()
    p8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(p8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def dice_iou_scores(logits, target):
    """Compute mean Dice and IoU after thresholding probabilities at 0.5."""
    prob = torch.sigmoid(logits)
    # thr = otsu_threshold(prob)
    thr = 0.5
    pred = (prob > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    dice = (2 * inter + EPS) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + EPS)
    iou = (inter + EPS) / (union + EPS)
    return dice.mean().item(), iou.mean().item()


class Down(nn.Module):
    """Encoder step: 2x downsampling via max-pool followed by conv block."""

    def __init__(self, in_ch, out_ch, norm="bn"):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = conv_block(in_ch, out_ch, norm=norm)

    def forward(self, x):
        """Apply pooling then feature extraction."""
        return self.block(self.pool(x))


class Up(nn.Module):
    """Decoder step: upsample, concatenate skip connection, then conv block."""

    def __init__(self, in_ch, out_ch, norm="bn", mode="bilinear"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.conv = conv_block(in_ch, out_ch, norm=norm)

    def forward(self, x, skip):
        """Fuse decoder feature map with its encoder skip feature map."""
        x = self.up(x)
        # Keep shapes aligned in case odd dimensions appear.
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def conv_block(in_ch, out_ch, norm="bn", p_drop=0.0):
    """
    A convolutional block of two convolutional layers with ReLU activation,
    batch normalization and optional dropout.

    Parameters:
    in_ch (int): Number of input channels
    out_ch (int): Number of output channels
    norm (str): Normalization type ("bn" for batch normalization, "gn" for group normalization)
    p_drop (float): Dropout probability (0.0 means no dropout)

    Returns:
    nn.Module: The convolutional block
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch) if norm=="bn" else nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch) if norm=="bn" else nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    ]
    if p_drop > 0:
        layers.insert(3, nn.Dropout2d(p_drop))  # between convs (optional)
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """Standard 2D U-Net for binary segmentation."""

    def __init__(self, in_ch=1, out_ch=1, base=32, norm="bn", p_drop_bottleneck=0.1):
        """Construct encoder/decoder stages and output projection head."""
        super().__init__()
        self.inc = conv_block(in_ch, base, norm=norm)
        self.down1 = Down(base, base * 2, norm=norm)
        self.down2 = Down(base * 2, base * 4, norm=norm)
        self.down3 = Down(base * 4, base * 8, norm=norm)
        self.down4 = Down(base * 8, base * 16, norm=norm)

        # Optional bottleneck dropout can regularize training.
        self.bott = conv_block(base * 16, base * 16, norm=norm, p_drop=p_drop_bottleneck)

        self.up4 = Up(in_ch=base * 16 + base * 8, out_ch=base * 8, norm=norm)
        self.up3 = Up(in_ch=base * 8 + base * 4, out_ch=base * 4, norm=norm)
        self.up2 = Up(in_ch=base * 4 + base * 2, out_ch=base * 2, norm=norm)
        self.up1 = Up(in_ch=base * 2 + base, out_ch=base, norm=norm)

        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming init for conv layers and unit init for normalization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Run encoder-decoder pass and return raw segmentation logits."""
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        b = self.bott(e5)
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        logits = self.outc(d1)
        return logits  # raw logits; apply sigmoid in loss/metrics
