# gui — Interactive 3D Column Viewer

Napari-based application for exploring 3D cell column labels, matching them with MINIAN calcium-imaging footprints, and exporting statistics from saved pairs.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point — parses CLI args, loads volume, launches Qt+Napari app |
| `core.py` | Volume I/O, overlap / adjacency math, MINIAN loading, PyVista plotting |
| `viewer.py` | `VolumeViewer` controller — wires Napari layers and all dock widgets |
| `widgets.py` | Five Qt dock widgets (see below) |
| `stats.py` | Batch CSV statistics for pairs and full-volume column summaries |

---

## Launching the Viewer

```bash
# From the TOOL/ root
python -m gui "G:\path\to\SG006_3D_D3"
```

The `data_dir` argument must point to a session folder that contains:

```
<data_dir>/
├── GLOBAL_PLANE_MASKS/
│   ├── plane_001_global.npy
│   ├── plane_002_global.npy
│   └── ...
├── PRE_PROCESSED_TIFF/
│   ├── <mouse_id>_P1.tif
│   └── ...
└── saved_column_pairs/       ← created automatically on first save
```

---

## Interactive Workflow

1. The **Full Stack** labels layer is added to Napari automatically.
2. **Double-click** a voxel to select that column label.
3. Use the dock widgets on the right:

| Widget | What it does |
|---|---|
| `CellListWidget` | Lists all non-zero label IDs; multi-select to filter the display |
| `NearbyWidget` | Shows overlapping and adjacent cells; click to add them to the selection |
| `MergeCellsWidget` | Merges two label IDs and optionally saves the edited volume |
| `LoadLayerWidget` | Picks a MINIAN folder and overlays footprint labels + image |
| `PairSaveWidget` | Saves, views, and deletes Minian-column pairs as `.npz` files |

4. The **Generate Stats** button calls `stats.generate_and_save_stats()` and writes a `column_metrics.csv` inside `saved_column_pairs/`.
5. The **Plot** button opens a PyVista 3D point-cloud view of the selected labels.

---

## Batch Statistics (`stats.py`)

Run directly to generate CSVs for multiple mice and sessions:

```bash
python -m gui.stats
```

Key functions:

- `minian_column_pair_stats(mouse_id)` — iterates all day × day Minian-column combinations, computes geometry + intensity metrics, writes per-combination and aggregated CSVs.
- `all_column_stats(mouse_id)` — loads the full global-mask volume per day, computes metrics for every non-zero label, writes day-level and all-days CSVs.
- `generate_and_save_stats(saved_pairs_dir, tiff_base_path, output_csv_path)` — single-session variant called by the GUI button.

### Metrics computed per column

| Metric | Description |
|---|---|
| `num_layers` | Number of planes containing the label |
| `start_layer` | First plane index |
| `total_volume` | Total voxel count |
| `50%_volume_idx` | Plane index at which 50 % of volume is accumulated |
| `max/min_plane_area` | Largest / smallest per-plane pixel count |
| `avg/max/min_intensity` | Aggregate TIFF intensity within the mask |
| `50%_intensity_layer` | Layer where 50 % of total intensity is accumulated |
| `max/min_API` | Max / min average-pixel intensity across layers |

---

## Data Conventions

- Volume shape: `(z, y, x)`, dtype `int`, background = `0`.
- Plane masks: `GLOBAL_PLANE_MASKS/plane_NNN_global.npy` (1-indexed, zero-padded to 3 digits).
- TIFF planes: `PRE_PROCESSED_TIFF/<mouse_id>_P<N>.tif` (1-indexed, no zero-padding).
- Pairs saved as `saved_column_pairs/<name>/<timestamp>.npz` with keys `minian_map`, `column_map`, `minian_id`, `column_id`.
- MINIAN Zarr folder must contain an `A` array with a `unit_id` dimension.

<!-- original content below preserved for reference -->
## What The Application Does

1. Loads per-plane global label masks (`plane_XXX_global.npy`) into one 3D volume.
2. Displays the full labeled stack in Napari.
3. Lets you:
   - select columns,
   - inspect overlapping and nearby labels,
   - load MINIAN footprints,
   - save Minian-column map pairs,
   - merge labels and export edited volume snapshots.
4. Computes offline stats from saved pairs and from all labels in a volume.

## Project Structure

- `main.py`: Entry point that loads one dataset and launches Napari.
- `core.py`: Core utilities for volume loading, overlap/adjacency detection, MINIAN loading, and PyVista plotting.
- `viewer.py`: Main controller (`VolumeViewer`) that wires Napari layers and Qt dock widgets.
- `widgets.py`: UI components (cell list, nearby list, merge/save, MINIAN loader, pair manager).
- `STATS.py`: Batch metrics pipeline for saved pairs and full-volume column summaries.

## Data Model And Assumptions

- Label volume shape is expected to be `(z, y, x)`.
- Background is label `0`.
- Plane files are expected as:
  - `GLOBAL_PLANE_MASKS/plane_001_global.npy`
  - `GLOBAL_PLANE_MASKS/plane_002_global.npy`
  - etc.
- TIFF planes are expected as:
  - `PRE_PROCESSED_TIFF/<mouse_id>_P1.tif`
  - `PRE_PROCESSED_TIFF/<mouse_id>_P2.tif`
  - etc.
- MINIAN data is loaded from a folder containing Zarr subdirectories and expects an `A` array with `unit_id`.

## Interactive Workflow

1. Start app with `python main.py`.
2. Main layer (`Full Stack`) is added as labels.
3. Use widgets on the right side:
   - `CellListWidget`: choose label IDs and filter display.
   - `NearbyWidget`: inspect overlap/adjacency computed by `near_by_cells`.
   - `MergeCellsWidget`: merge label IDs and optionally save edited volume.
   - `LoadLayerWidget`: load MINIAN layer and overlay footprint labels.
   - `PairSaveWidget`: save/view/delete Minian-column pairs as compressed `.npz`.
4. Optional PyVista point visualization is available via the `Plot` button.

## Batch Statistics Workflow (`STATS.py`)

Primary routines:

- `minian_column_pair_stats(mouse_id)`:
  - iterates day/day combinations,
  - loads saved pair files,
  - computes geometry + intensity metrics,
  - writes per-combination CSV and final aggregated CSV.

- `all_column_stats(mouse_id)`:
  - loads full global-mask volume per day,
  - computes metrics for each non-zero label,
  - writes day-level CSV and final all-days CSV.

- `save_layer_metrics_column(minian_id, session, mouse_id)`:
  - saves per-layer area/intensity values for one chosen Minian ID.

## Dependencies

Install (or ensure availability of) at least:

- `numpy`
- `scipy`
- `xarray`
- `dask`
- `pyvista`
- `matplotlib`
- `napari`
- `PyQt5`
- `pandas`
- `tifffile`

## Critical Review Notes

1. `viewer.py` imports `generate_and_save_stats` from `stats`, but the repository currently contains `STATS.py` and does not define `generate_and_save_stats`. The `Generate Stats` button path should be aligned with an existing function.
2. Paths are hard-coded to `G:\AK\Cell_Overlap\New\...` in multiple places (`main.py`, `STATS.py`), reducing portability.
3. Most workflows rely on strict file naming conventions; missing or mismatched names silently skip data in some functions.
4. Several methods use direct `print` logging; replacing with structured logging would help debugging and reproducibility.

## Recommended Next Improvements

1. Add a single config file (or CLI args) for `main_dir`, `mouse_id`, and day/session selections.
2. Standardize stats module naming (`stats.py` vs `STATS.py`) and expose one clear API function for UI-triggered stats export.
3. Add lightweight tests for:
   - `load_volume`,
   - `near_by_cells`,
   - `compute_column_metrics`,
   - `compute_layer_intensity_metrics`.
