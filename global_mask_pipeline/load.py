import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np


def _normalize_cell_id(cell_id):
    """Normalize cell ids so file lookup and chain logic use one representation."""
    try:
        return str(int(cell_id))
    except Exception:
        return str(cell_id)


def load_preprocessed_layer(layer_dir):
    """Load centroid table and KD-tree for one processed plane directory."""
    layer_path = Path(layer_dir)

    centroid_files = sorted(layer_path.glob("*_centroids.csv"))
    if not centroid_files:
        raise FileNotFoundError(f"No centroid CSV found in {layer_dir}")
    centroid_file = centroid_files[0]

    centroids = {}
    with open(centroid_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = _normalize_cell_id(row["cell_id"])
            centroids[cell_id] = (float(row["centroid_x"]), float(row["centroid_y"]))

    kd_files = sorted(layer_path.glob("*_kd_tree.pkl"))
    if not kd_files:
        raise FileNotFoundError(f"No KD-tree pickle found in {layer_dir}")
    kd_file = kd_files[0]

    with open(kd_file, "rb") as f:
        cell_ids, tree = pickle.load(f)

    cell_ids = [_normalize_cell_id(cid) for cid in cell_ids]
    return {"centroids": centroids, "kd_tree": (cell_ids, tree)}


def get_candidate_cells(curr_centroid, kd_tree_data, radius=20):
    """Return nearby candidate cell ids in the next layer within a radius."""
    cell_ids, tree = kd_tree_data
    point = np.asarray(curr_centroid, dtype=float)
    idxs = tree.query_ball_point(point, r=radius)
    return [cell_ids[i] for i in idxs]


def load_mask_for_cell(layer_dir, cell_id):
    """Load one cell mask from .npy/.npz/.json exports created by cell_processor.py."""
    layer_path = Path(layer_dir)
    cid = _normalize_cell_id(cell_id)

    npy_path = layer_path / f"cell_{cid}.npy"
    if npy_path.exists():
        return np.load(npy_path).astype(bool)

    npz_path = layer_path / f"cell_{cid}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if "mask" in data:
            return data["mask"].astype(bool)
        keys = list(data.keys())
        if not keys:
            raise ValueError(f"Empty NPZ file: {npz_path}")
        return data[keys[0]].astype(bool)

    json_path = layer_path / f"cell_{cid}.json"
    if json_path.exists():
        with open(json_path) as f:
            return np.asarray(json.load(f)).astype(bool)

    raise FileNotFoundError(
        f"No mask file found for cell {cid} in {layer_dir}. "
        "Expected .npy, .npz, or .json."
    )
