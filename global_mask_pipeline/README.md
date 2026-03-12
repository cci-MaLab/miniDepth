# global_mask_pipeline

Utilities for converting per-plane labeled masks into globally consistent plane masks by linking the same cell across neighboring depths.

This part of the project is a post-processing pipeline. It does not train a model. Instead, it takes labeled plane masks that already exist, breaks them into per-cell artifacts, links matching cells across planes, and writes `GLOBAL_PLANE_MASKS/plane_XXX_global.npy` outputs that the GUI and downstream analysis can consume.

## What This Pipeline Does

The pipeline has four stages:

1. Read each labeled plane TIFF and split it into one binary mask per cell.
2. Compute a centroid for every cell and build a KD-tree for fast spatial lookup.
3. Link cells across neighboring planes into chains using centroid proximity plus mask overlap.
4. Assign one global ID per chain and export plane-wise global label images.

The top-level runner is [save_global_plane_masks.py](save_global_plane_masks.py).

## Why The KD-tree Is Important

The KD-tree is the candidate-pruning step.

Without it, each cell in one plane would need to be compared against every cell in the next plane. That becomes expensive quickly when each plane has many labels. Instead, the code:

- computes centroids for all cells in a plane,
- stores them as `(cell_id, x, y)` entries,
- builds a `scipy.spatial.cKDTree` from the `(x, y)` coordinates,
- queries only nearby cells in the next plane.

In the current implementation, the radius search is done in [load.py](load.py) via `get_candidate_cells(...)`, and [chains.py](chains.py) calls it with a radius of `20` pixels.

So the KD-tree is not the matcher itself. It is the fast pre-filter that limits the overlap test to a small spatial neighborhood.

## Files

| File | Role |
|---|---|
| [cell_processor.py](cell_processor.py) | Splits each labeled plane into per-cell masks, computes centroids, optionally builds a KD-tree |
| [load.py](load.py) | Loads centroid CSVs, KD-tree pickles, and individual cell masks |
| [chains.py](chains.py) | Builds cross-plane chains using KD-tree candidates and overlap checks |
| [global_masks.py](global_masks.py) | Converts chains into global IDs and exports `plane_XXX_global.npy` files |
| [save_global_plane_masks.py](save_global_plane_masks.py) | End-to-end script for one dataset directory |

## Detailed Pipeline

### 1. Per-cell extraction and centroid computation

[cell_processor.py](cell_processor.py) reads each labeled TIFF plane and does the following:

- loads the labeled image,
- finds all unique labels except background `0`,
- creates a binary mask for each label,
- computes a centroid using `cv2.moments`,
- saves each mask as `cell_<id>.npy` by default,
- writes one centroid CSV per plane,
- optionally builds one KD-tree pickle per plane.

Centroids are saved as:

```text
cell_id, centroid_x, centroid_y
```

The centroid is computed from the binary mask itself, not from a bounding box.

### 2. KD-tree creation

Still in [cell_processor.py](cell_processor.py), `build_and_save_kd_tree(...)` stores a pickle containing:

- the ordered list of cell IDs,
- the `cKDTree` built from `(centroid_x, centroid_y)`.

This is saved as `*_kd_tree.pkl` inside each processed plane folder.

### 3. Cross-plane chaining

[chains.py](chains.py) is where plane-to-plane linking happens.

The chaining logic is forward-only and local:

- processed plane folders are collected and sorted by plane number,
- for each unvisited cell in a plane, the code tries to extend a chain into the next plane,
- KD-tree search returns nearby candidate cells in the next plane,
- each candidate is then validated by binary-mask overlap,
- the first acceptable candidate is taken as the match,
- once a cell misses in the next plane, the chain stops.

### 4. Match criterion

Two binary masks are compared by directional overlap:

- `overlap_percent1 = intersection / area(mask1)`
- `overlap_percent2 = intersection / area(mask2)`

A pair is considered a match when either direction exceeds the threshold.

Current default in [chains.py](chains.py):

- overlap threshold: `0.6`
- KD-tree search radius: `20`

This means matching is based on both:

- spatial proximity, via the KD-tree,
- shape agreement, via overlap percentage.

### 5. Global ID assignment

[global_masks.py](global_masks.py) enumerates chains starting at `1` and assigns one global ID per chain.

For example, if a chain is:

```python
{"P1": 12, "P2": 7, "P3": 4}
```

then all those local cell IDs are mapped to the same global ID.

The code then rebuilds each plane image by reading the saved per-cell masks and writing the global ID into the output array.

Outputs are saved as:

- `plane_001_global.npy`
- `plane_002_global.npy`
- ...

## Input Layout

Point the runner at a directory like:

```text
<Labeled_Masks>/
├── <something>_P1.tif
├── <something>_P2.tif
├── <something>_P3.tif
└── ...
```

Important naming behavior:

- [cell_processor.py](cell_processor.py) uses the last underscore-separated token in the filename stem as the plane name.
- So a filename like `SG006_D7_P1.tif` becomes plane folder `P1`.
- The later chaining code expects processed plane folders named like `P1`, `P02`, or `P003`.

## Generated Intermediate Layout

After `process_tiff_directory(...)`, the folder looks like:

```text
<Labeled_Masks>/PROCESSED/
├── P1/
│   ├── cell_1.npy
│   ├── cell_2.npy
│   ├── P1_centroids.csv
│   └── P1_kd_tree.pkl
├── P2/
│   ├── cell_3.npy
│   ├── P2_centroids.csv
│   └── P2_kd_tree.pkl
└── ...
```

## Final Outputs

[save_global_plane_masks.py](save_global_plane_masks.py) writes:

- a chain pickle file such as `SG006_3D_D7_chains.pkl`
- a `GLOBAL_PLANE_MASKS/` directory containing:

```text
<Labeled_Masks>/GLOBAL_PLANE_MASKS/
├── plane_001_global.npy
├── plane_002_global.npy
├── plane_003_global.npy
└── ...
```

Each output array is a 2D integer image where:

- `0` = background or unmapped cell
- `1, 2, 3, ...` = global IDs assigned from chains

## Run

From the repository root:

```bash
python -m global_mask_pipeline.save_global_plane_masks
```

Before running, edit these constants in [save_global_plane_masks.py](save_global_plane_masks.py):

- `TIFF_DIR`: folder containing the labeled plane TIFF files
- `PKL_FILE`: output pickle file for chains
- `NUM_PLANES`: number of planes to export

## Current Assumptions In The Code

These are important because they affect output behavior:

- Background label is `0`.
- Matching is only attempted between consecutive planes.
- The algorithm does not skip missing planes.
- The algorithm stops a chain at the first unmatched step.
- Candidate search uses centroid proximity only; there is no appearance model.
- Match validation uses overlap only; there is no learned similarity score.
- Only cells that belong to a chain receive a global ID.
- Unchained cells fall back to `0` in the exported global mask.

## Collision And Overlap Handling

[global_masks.py](global_masks.py) reports two kinds of issues:

1. Chain collisions

- If the same `(plane_idx, local_id)` appears in more than one chain, the code records it as a collision.

2. Pixel overlaps while rebuilding a plane

- If two saved binary masks write into the same output pixels, those pixels are counted as overlaps.
- The current policy is effectively last-write-wins.

So the pipeline does not silently ignore everything. It reports these conditions, but it does not yet resolve them with a more advanced reconciliation strategy.

## Practical Limitations

This is a simple and useful pipeline, but it has clear limits:

- It assumes cells move smoothly enough across planes that centroid radius search is meaningful.
- It assumes overlap between neighboring-plane masks is strong enough for a fixed threshold.
- It does not handle splits or merges explicitly.
- It does not search backward once a forward match fails.
- It does not currently preserve isolated single-plane cells as unique global IDs.

If those behaviors become important, the next step would be to add:

- configurable CLI arguments,
- configurable radius and overlap threshold,
- support for plane skipping,
- branch/split handling,
- optional retention of unchained cells as their own global IDs.

## Suggested Usage

Use this pipeline when:

- you already have labeled masks per plane,
- you want one stable ID for the same cell across depths,
- you need `GLOBAL_PLANE_MASKS` for the viewer or downstream statistics.

Do not use it as a replacement for segmentation itself. It is a post-segmentation linking/export stage.
