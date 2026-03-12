import numpy as np
import os
import re
from collections import defaultdict
PLANE_PREFIX = "P"
MASK_EXT = ".npy"

def extract_int(filename):
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else None


def build_global_from_npy_masks(mapping, plane_idx, plane_folder, out_path):
    """
    Compose global label image from per-cell mask .npy files in plane_folder.
    mapping keys use ints for plane_idx and local_id.
    """
    npy_files = sorted([f for f in os.listdir(plane_folder) if f.lower().endswith(MASK_EXT)])
    if not npy_files:
        return None

    # load first mask to get image size
    sample = np.load(os.path.join(plane_folder, npy_files[0]))
    H, W = sample.shape
    global_img = np.zeros((H, W), dtype=np.uint32)

    overlaps = 0
    for fname in npy_files:
        lid = extract_int(fname)
        if lid is None:
            print("Skipping unparseable file:", fname); continue
        mask = np.load(os.path.join(plane_folder, fname)).astype(bool)
        key = (plane_idx, int(lid))
        gid = mapping.get(key, 0)  # default 0 = background/unmapped
        # detect overlap (non-zero pixels already assigned)
        overlap_pixels = np.count_nonzero(global_img[mask] != 0)
        if overlap_pixels:
            overlaps += overlap_pixels
        # assign global id (last write wins if conflicts)
        if gid != 0:
            global_img[mask] = gid

    # save
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, f"plane_{plane_idx:03d}_global.npy"), global_img)
    
    print(f"Plane {plane_idx:03d}: {len(npy_files)} masks, {overlaps} overlaps")
    return overlaps

def build_map_from_chains(chains):
    """
    chains: list of chains. Each chain can be:
      - a dict like {'P1': 2, 'P2': 3, ...}
      - or a tuple/list aligned with plane numbers (less common)
      - or you can adapt input format here.
    Returns:
      mapping: dict (plane_idx:int, local_id:int) -> global_id:int
      collisions: dict (plane_idx, local_id) -> list of global_ids (if collision)
    """
    mapping = {}
    collisions = defaultdict(list)
    for gid, chain in enumerate(chains, start=1):
        # If chain is a dict mapping plane-name -> local_id:
        if isinstance(chain, dict):
            for plane_name, local in chain.items():
                pidx = extract_int(plane_name)
                if pidx is None:
                    raise ValueError(f"Can't parse plane from {plane_name}")
                key = (int(pidx), int(local))
                if key in mapping and mapping[key] != gid:
                    collisions[key].append(mapping[key])
                    collisions[key].append(gid)
                else:
                    mapping[key] = gid
        # else if chain is a list of local ids for consecutive planes starting at some known plane,
        # adapt this block accordingly.
        else:
            raise ValueError("Chains must be dicts mapping plane name -> local id.")
    # normalize collisions lists (unique)
    for k in list(collisions.keys()):
        collisions[k] = sorted(set(collisions[k]))
    return mapping, collisions

def create_global_planes(chains, planes_root, out_dir, num_planes):

    mapping, collisions = build_map_from_chains(chains)

    if collisions:
        print(f"WARNING: {len(collisions)} collisions detected (same local cell in multiple chains).")
        # print some example collisions
        for i, (k, owners) in enumerate(collisions.items()):
            print(" Collision:", k, "owners:", owners)

    os.makedirs(out_dir, exist_ok=True)
    total_overlaps = 0
    for p in range(1, num_planes + 1):
        # try to find plane folder (common names)
        candidates = [
            os.path.join(planes_root, f"{PLANE_PREFIX}{p}"),
            os.path.join(planes_root, f"{PLANE_PREFIX}{p:02d}"),
            os.path.join(planes_root, f"{PLANE_PREFIX}{p:03d}"),
        ]
        plane_folder = None
        for c in candidates:
            if os.path.isdir(c):
                plane_folder = c
                break

        if plane_folder is None:
            # maybe you have a single local_label_image file instead
            local_label_path = os.path.join(planes_root, f"layer_{p:03d}_local.npy")

        # Prefer per-cell masks if present
        overlaps = build_global_from_npy_masks(mapping, p, plane_folder, out_dir)
        if overlaps is not None:
            total_overlaps += overlaps

    print("Done. total overlapped pixels:", total_overlaps)
    return mapping, collisions