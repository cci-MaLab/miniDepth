"""Inference and evaluation helpers for patch-wise U-Net prediction.

Includes tiled probability-map inference, thresholding, metric computation,
and multi-threshold evaluation over full images.
"""

import numpy as np, torch, cv2, pandas as pd
from pathlib import Path
from .model import UNet
from torchsummary import summary
from typing import Sequence
from typing import Optional
import matplotlib.pyplot as plt
import tifffile


def load_ckpt(path, model, optimizer=None, scheduler=None, map_location="cuda"):
    """Load checkpoint weights and optional optimizer/scheduler state."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)

def load_model(path):
    """Instantiate U-Net and load model weights from checkpoint path."""
    model = UNet(in_ch=1, out_ch=1, base=32, norm="bn").cuda()
    load_ckpt(path, model)
    return model


@torch.no_grad()
def predict_prob_map(model, img, tile=256, overlap=64, device="cuda"):
    """Predict full-image probability map using overlap-and-average tiling."""
    model.eval()
    H, W = img.shape
    prob = np.zeros((H, W), np.float32)
    cnt  = np.zeros((H, W), np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    step = tile - overlap
    for y in range(0, H, step):
        for x in range(0, W, step):
            patch = img[y:y+tile, x:x+tile]
            ph, pw = patch.shape
            if ph < tile or pw < tile:
                # Reflect padding keeps edge context smoother than zero padding.
                patch = np.pad(patch, ((0, tile-ph), (0, tile-pw)), mode="reflect")
            t = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)
            p = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()[:ph, :pw]
            prob[y:y+ph, x:x+pw] += p
            cnt[y:y+ph, x:x+pw]  += 1
    prob /= np.maximum(1, cnt)
    return prob

def otsu_threshold(prob: np.ndarray) -> float:
    """Compute Otsu threshold on probability map and return value in [0, 1]."""
    p8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(p8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def dice_iou_from_binary(pred_bool, gt_bool, eps=1e-6):
    """Return Dice/IoU and confusion counts for binary prediction and GT."""
    pred_bool = pred_bool.astype(bool)
    gt_bool   = gt_bool.astype(bool)

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()
    tn = np.logical_and(~pred_bool, ~gt_bool).sum()

    dice = (2*tp + eps) / (2*tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)

    return float(dice), float(iou), int(tp), int(fp), int(fn), int(tn)

def nearest_train_distance(plane_idx: int, train_planes: Sequence[int]) -> int:
    """Return absolute distance to the nearest training plane index."""
    return int(min(abs(plane_idx - t) for t in train_planes))


def threshold_sweep(prob: np.ndarray, gt_bool: np.ndarray, sweep_list: Sequence[float]):
    """Evaluate multiple thresholds and return the best-by-Dice candidate.

    Returns:
        tuple: (best_dice, best_threshold, rows)
            rows is a list of per-threshold metric tuples:
            (threshold, dice, iou, tp, fp, fn, tn).
    """
    best_dice = -1.0
    best_thr = float(sweep_list[0])
    rows = []

    for thr in sweep_list:
        thr = float(thr)
        pred = prob >= thr
        d, i, tp, fp, fn, tn = dice_iou_from_binary(pred, gt_bool)
        rows.append((thr, d, i, tp, fp, fn, tn))
        if d > best_dice:
            best_dice = d
            best_thr = thr

    return best_dice, best_thr, rows


def _save_overlay(
    folder: Path,
    plane_idx: int,
    img: np.ndarray,
    pred_bool: np.ndarray,
    gt_bool: Optional[np.ndarray] = None,
):
    """Save qualitative overlay and binary mask for one plane prediction."""
    folder.mkdir(parents=True, exist_ok=True)
    vis = (np.clip(img / (img.max() + 1e-8) * 255, 0, 255)).astype(np.uint8)
    overlay = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if gt_bool is None:
        # old behavior: all predicted positives in green
        overlay[pred_bool > 0] = (0, 255, 0)
    else:
        tp = np.logical_and(pred_bool, gt_bool)
        fp = np.logical_and(pred_bool, ~gt_bool)
        fn = np.logical_and(~pred_bool, gt_bool)

        overlay[tp] = (0, 255, 0)     # green = true positive
        overlay[fp] = (0, 0, 255)     # red = false positive
        overlay[fn] = (255, 0, 0)     # blue = false negative

    cv2.imwrite(str(folder / f"plane_{plane_idx}.png"), overlay)

    # Save binary mask too
    mask_dir = folder / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(mask_dir / f"plane_{plane_idx}.tiff"), pred_bool.astype(np.uint8) * 255)



def eval_full_images_multi(
    df_subset: pd.DataFrame,
    model,
    modes=("otsu","global","sweep"),
    global_thr: Optional[float] = None,
    sweep_list: Optional[Sequence[float]] = None,
    save_overlays_root: Optional[str] = None,
    train_planes: Optional[Sequence[int]] = None,
    dataset_name: str = "test",
    device="cuda"
) -> pd.DataFrame:
    """
    Evaluate multiple threshold modes in one pass.
      modes: any of {"otsu","global","sweep"}
      global_thr: scalar used when "global" in modes
      sweep_list: iterable of thresholds when "sweep" in modes
    Returns a tidy DataFrame with one row per (plane, mode, threshold).
    """
    rows = []
    outroot = Path(save_overlays_root) if save_overlays_root else None
    if outroot: outroot.mkdir(parents=True, exist_ok=True)

    for _, row in df_subset.iterrows():
        # ---- load data
        img = cv2.imread(row.image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        msk = cv2.imread(row.mask_path, cv2.IMREAD_UNCHANGED)
        if msk.ndim == 3: msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        gt = (msk > 0)

        plane = int(row.plane_idx)
        dz = nearest_train_distance(plane, train_planes) if train_planes is not None else None

        # ---- prob map once
        prob = predict_prob_map(model, img, device=device)


        # ---- evaluate each mode
        for mode in modes:
            if mode == "otsu":
                thr = otsu_threshold(prob)
                pred = (prob >= thr)
                label = "otsu"
                d, i, tp, fp, fn, tn = dice_iou_from_binary(pred, gt)
                rows.append(dict(dataset=dataset_name, plane_idx=plane, dz=dz,
                                 mode=label, threshold=float(thr), dice=d, iou=i,
                                 tp=tp, fp=fp, fn=fn, tn=tn))

                if outroot:
                    _save_overlay(outroot/label, plane, img, pred)

            elif mode == "global":
                assert global_thr is not None, "global_thr must be provided for mode='global'"
                thr = float(global_thr)
                pred = (prob >= thr)
                label = f"global_{thr:.3f}"
                d, i, tp, fp, fn, tn = dice_iou_from_binary(pred, gt)
                rows.append(dict(dataset=dataset_name, plane_idx=plane, dz=dz,
                                 mode=label, threshold=float(thr), dice=d, iou=i,
                                 tp=tp, fp=fp, fn=fn, tn=tn))
                if outroot:
                    _save_overlay(outroot/label, plane, img, pred)

            elif mode == "sweep":
                assert sweep_list is not None and len(sweep_list) > 0, "Provide sweep_list for mode='sweep'"
                for thr in sweep_list:
                    thr = float(thr)
                    pred = (prob >= thr)
                    d, i, tp, fp, fn, tn = dice_iou_from_binary(pred, gt)
                    label = f"sweep"
                    rows.append(dict(dataset=dataset_name, plane_idx=plane, dz=dz,
                                     mode=label, threshold=thr, dice=d, iou=i,
                                     tp=tp, fp=fp, fn=fn, tn=tn))
                if outroot:
                    # also save best-at-sweep overlay for quick glance
                    best_d, best_t, _ = threshold_sweep(prob, gt, sweep_list)
                    _save_overlay(outroot/("sweep_best"), plane, img, (prob >= best_t))
            else:
                raise ValueError(f"Unknown mode: {mode}")

    return pd.DataFrame(rows)