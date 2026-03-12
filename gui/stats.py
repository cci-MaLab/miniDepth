"""Batch statistics utilities for column and Minian-column pair analyses.

This script computes geometry and intensity metrics from saved pair maps and
from full global-mask volumes, and writes CSV summaries.
"""

import numpy as np
import glob
import csv
import os
from datetime import datetime

import pandas as pd
import tifffile

from .core import load_volume


def load_saved_pair(pair_path):
    """Load a saved Minian-column pair `.npz` file and log key metadata."""
    data = np.load(pair_path)
    print(f"Loaded pair from {pair_path}")
    print(f"Minian map shape: {data['minian_map'].shape}")
    print(f"Column map shape: {data['column_map'].shape}")
    print(f"Minian ID: {data['minian_id']}, Column ID: {data['column_id']}")
    return data


def compute_column_metrics(column_map):
    """Compute geometry-centric metrics for a single 3D column map.

    Args:
        column_map: 3D array where non-zero values mark column voxels.

    Returns:
        Tuple containing layer/volume metrics and names for max/min area planes.
    """
    print("Computing column metrics")
    layer_pixel_counts = (column_map > 0).sum(axis=(1, 2))
    print(f"Layer pixel counts: {layer_pixel_counts}")
    num_layers = np.count_nonzero(layer_pixel_counts)
    print(f"Number of layers: {num_layers}")
    nonzero_layers = np.where(layer_pixel_counts > 0)[0]
    print(f"Nonzero layers: {nonzero_layers}")
    start_layer = nonzero_layers[0] if nonzero_layers.size > 0 else None
    print(f"Starting layer: {start_layer}")
    total_volume = layer_pixel_counts.sum()
    print(f"Total volume: {total_volume}")

    if layer_pixel_counts.size and layer_pixel_counts.max() > 0:
        max_plane_idx = int(np.argmax(layer_pixel_counts)) + 1
        max_plane_area = int(layer_pixel_counts[max_plane_idx - 1])
        max_area_plane_name = f"P{max_plane_idx:02d}"
        print(f"Max plane index: {max_plane_idx}, Max plane area: {max_plane_area}")
    else:
        max_plane_idx = None
        max_plane_area = 0

    # Min area is computed only across non-empty planes.
    positive_mask = layer_pixel_counts > 0
    if positive_mask.any():
        masked = np.where(positive_mask, layer_pixel_counts, np.inf)
        min_plane_idx = int(np.argmin(masked))
        min_plane_area = int(layer_pixel_counts[min_plane_idx])
        min_area_plane_name = f"P{min_plane_idx+1:02d}"
        print(f"Min plane index: {min_plane_idx}, Min plane area: {min_plane_area}")
    else:
        min_plane_idx = None
        min_plane_area = 0
        min_area_plane_name = None

        
    cumsum = np.cumsum(layer_pixel_counts)
    print(f"Cumulative sum: {cumsum}")
    if total_volume > 0:
        fifty_percent_idx = np.searchsorted(cumsum, total_volume / 2)
        print(f"50% volume index: {fifty_percent_idx}")
        fifty_percent_depth = fifty_percent_idx
    else:
        fifty_percent_depth = None
    print(f"50% volume index: {fifty_percent_depth}")

    return (
        num_layers,
        start_layer,
        total_volume,
        fifty_percent_depth,
        nonzero_layers,
        max_plane_area,
        max_area_plane_name,
        min_plane_area,
        min_area_plane_name,
    )


def compute_layer_intensity_metrics(column_map, nonzero_layers, base_path, mouse_id):
    """Compute per-layer and aggregate intensity metrics from TIFF planes.

    Args:
        column_map: 3D array where non-zero voxels define the column mask.
        nonzero_layers: Array of layer indices where the mask is present.
        base_path: Directory containing per-plane TIFF files.
        mouse_id: Prefix used in TIFF naming convention.

    Returns:
        Tuple of aggregate and layer-indexed intensity metrics.
    """
    intensities = []
    layer_names = []
    num_pixels_per_layer = []
    avg_pixel_intensity_per_layer = []
    for idx in nonzero_layers:
        layer_num = idx
        plane_file_name_tif = f"{mouse_id}_P{layer_num+1:01d}.tif"
        tiff_path = os.path.join(base_path, plane_file_name_tif)
        if not os.path.exists(tiff_path):
            print(f"File not found: {tiff_path}")
            intensities.append(np.nan)
            layer_names.append(f"P{layer_num+1}")
            continue
        print(f"Reading file: {tiff_path}")
        img = tifffile.imread(tiff_path)
        mask = column_map[idx] > 0
        intensity = img[mask].sum()
        intensities.append(intensity)
        layer_names.append(f"P{layer_num+1}")

        num_pixels = np.sum(mask)
        avg_pixel_intensity = intensity / num_pixels if num_pixels > 0 else 0
        num_pixels_per_layer.append(num_pixels)
        avg_pixel_intensity_per_layer.append(avg_pixel_intensity)

    # API = average pixel intensity within masked pixels for each layer.
    layer_max_avg_pxl_intensity = np.max(avg_pixel_intensity_per_layer)
    layer_idx_max_avg_pxl_intensity = np.argmax(avg_pixel_intensity_per_layer)

    layer_min_avg_pxl_intensity = np.min(avg_pixel_intensity_per_layer)
    layer_idx_min_avg_pxl_intensity = np.argmin(avg_pixel_intensity_per_layer)

    fifty_percent_api = np.cumsum(avg_pixel_intensity_per_layer)
    print(f"Cumulative avg pixel intensity: {fifty_percent_api}")
    fifty_percent_api_idx = np.searchsorted(fifty_percent_api, np.array(avg_pixel_intensity_per_layer).sum() / 2)

    print(f"Intensities: {intensities}")
    intensities = np.array(intensities)
    valid = ~np.isnan(intensities)
    if valid.any():
        avg_intensity = intensities[valid].mean()
        max_intensity = intensities[valid].max()
        min_intensity = intensities[valid].min()
        max_layer = layer_names[np.argmax(intensities[valid])]
        min_layer = layer_names[np.argmin(intensities[valid])]

        total_intensity = intensities[valid].sum()
        print(f"Total intensity: {total_intensity}")
        if total_intensity > 0:
            cumsum_intensity = np.cumsum(intensities[valid])
            rel_idx = np.searchsorted(cumsum_intensity, total_intensity / 2)
            fifty_percent_intensity_layer = int(nonzero_layers[valid][rel_idx]) + 1
            print(f"50% intensity layer: P{fifty_percent_intensity_layer:02d}")
            fifty_percent_intensity_layer_name = f"P{fifty_percent_intensity_layer:02d}"

        else:
            fifty_percent_intensity_layer_name = None
            fifty_percent_intensity_layer = None
    else:
        avg_intensity = max_intensity = min_intensity = max_layer = min_layer = None
        layer_max_avg_pxl_intensity = layer_idx_max_avg_pxl_intensity = None
        layer_min_avg_pxl_intensity = layer_idx_min_avg_pxl_intensity = None

    print(f"Average intensity: {avg_intensity}")
    print(f"Max intensity: {max_intensity} at {max_layer}")
    print(f"Min intensity: {min_intensity} at {min_layer}")
    print(f"50% intensity layer: {fifty_percent_intensity_layer_name}")
    return (
        avg_intensity,
        max_intensity,
        max_layer,
        min_intensity,
        min_layer,
        fifty_percent_intensity_layer_name,
        layer_max_avg_pxl_intensity,
        layer_idx_max_avg_pxl_intensity,
        layer_min_avg_pxl_intensity,
        layer_idx_min_avg_pxl_intensity,
        fifty_percent_api_idx,
    )



def minian_column_pair_stats(mouse_id):
    """Generate CSV stats for all Minian-session and column-day combinations."""
    main_dir = r"G:\AK\Cell_Overlap\New"
    cols = ["D0", "D1", "D3", "D7"]
    minian = ["D0", "D1", "D3", "D7"]
    df = []
    for col in cols:
        for m in minian:
            print(f"Processing Minian {m} and Column {col}")
            saved_pairs_dir = os.path.join(main_dir, f"{mouse_id}_3D_{col}", "saved_column_pairs", f"Minian_{m}_Col_{col}")
            output_csv_path = os.path.join(saved_pairs_dir, f"Minian_{m}_Col_{col}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
            tiff_base_path = os.path.join(main_dir, f"{mouse_id}_3D_{col}", "PRE_PROCESSED_TIFF")
            npz_files = glob.glob(os.path.join(saved_pairs_dir, "*.npz"))
            if not npz_files:
                print(f"No .npz files found in the directory: {saved_pairs_dir}")
                continue
            
            
            with open(output_csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["minian", "Col",
                    "minian_id", "column_id", "num_layers", "start_layer", "total_volume", "50%_percent_volume_idx",
                    "max_plane_area", "max_plane_area_idx", "min_plane_area", "min_plane_area_idx",
                    "avg_intensity", "max_intensity", "max_intensity_idx", "min_intensity", "min_intensity_idx",
                    "50%_intensity_idx", "max_API", "max_API_idx", "min_API", "min_API_idx", "50%_API_idx"
                ])
                for pair_path in npz_files:
                    print(f"Processing file: {pair_path}")
                    data = load_saved_pair(pair_path)
                    column_map = data['column_map']
                    minian_id = data['minian_id']
                    column_id = data['column_id']

                    (
                        num_layers,
                        start_layer,
                        total_volume,
                        fifty_percent_depth,
                        nonzero_layers,
                        max_plane_area,
                        max_area_plane_name,
                        min_plane_area,
                        min_area_plane_name,
                    ) = compute_column_metrics(column_map)
                    (
                        avg_intensity,
                        max_intensity,
                        max_layer,
                        min_intensity,
                        min_layer,
                        fifty_percent_intensity_layer,
                        layer_max_avg_pxl_intensity,
                        layer_idx_max_avg_pxl_intensity,
                        layer_min_avg_pxl_intensity,
                        layer_idx_min_avg_pxl_intensity,
                        fifty_percent_api_idx,
                    ) = compute_layer_intensity_metrics(column_map, nonzero_layers, tiff_base_path, mouse_id)
                    
                    writer.writerow([m, col,
                        minian_id, column_id, num_layers, start_layer, total_volume, fifty_percent_depth,
                        max_plane_area, max_area_plane_name, min_plane_area, min_area_plane_name,
                        avg_intensity, max_intensity, max_layer, min_intensity, min_layer, fifty_percent_intensity_layer, layer_max_avg_pxl_intensity, start_layer + layer_idx_max_avg_pxl_intensity, layer_min_avg_pxl_intensity, start_layer + layer_idx_min_avg_pxl_intensity, start_layer + fifty_percent_api_idx
                    ])
                    df.append([m, col,
                        minian_id, column_id, num_layers, start_layer, total_volume, fifty_percent_depth,
                        max_plane_area, max_area_plane_name, min_plane_area, min_area_plane_name,
                        avg_intensity, max_intensity, max_layer, min_intensity, min_layer, fifty_percent_intensity_layer, layer_max_avg_pxl_intensity, start_layer + layer_idx_max_avg_pxl_intensity, layer_min_avg_pxl_intensity, start_layer + layer_idx_min_avg_pxl_intensity, start_layer + fifty_percent_api_idx
                    ])
            print(f"Finished processing Minian {m} and Column {col}")
            print(f"Stats saved to {output_csv_path}")

    final_df = pd.DataFrame(df, columns=["minian", "Col",
        "minian_id", "column_id", "num_layers", "start_layer", "total_volume", "50%_percent_volume_idx",
        "max_plane_area", "max_plane_area_idx", "min_plane_area", "min_plane_area_idx",
        "avg_intensity", "max_intensity", "max_intensity_idx", "min_intensity", "min_intensity_idx",
        "50%_intensity_idx", "max_API", "max_API_idx", "min_API", "min_API_idx", "50%_API_idx"
    ])
    final_df.to_csv(f"{mouse_id}_stats_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)


def all_column_stats(mouse_id):
    """Generate stats for every labeled column in each day for one mouse."""

    main_dir = r"G:\AK\Cell_Overlap\New"
    cols = ["D0", "D1", "D3", "D7"]
    all_df = []
    for col in cols:
        df = []
        print(f"Processing Column {col}")
        global_masks_dir = os.path.join(main_dir, f"{mouse_id}_3D_{col}", "GLOBAL_PLANE_MASKS")

        output_csv_path = os.path.join(main_dir, f"{mouse_id}_3D_{col}", f"All_Minian_Col_{col}.csv")
        tiff_base_path = os.path.join(main_dir, f"{mouse_id}_3D_{col}", "PRE_PROCESSED_TIFF")

        volume = load_volume(global_masks_dir, len(os.listdir(global_masks_dir)))

        uniq_labels = np.unique(volume)

        for label in uniq_labels:
            if label == 0:
                continue
            print(f"Processing label: {label}")
            column_map = (volume == label).astype(np.uint8)
            (
                num_layers,
                start_layer,
                total_volume,
                fifty_percent_depth,
                nonzero_layers,
                max_plane_area,
                max_area_plane_name,
                min_plane_area,
                min_area_plane_name,
            ) = compute_column_metrics(column_map)
            (
                avg_intensity,
                max_intensity,
                max_layer,
                min_intensity,
                min_layer,
                fifty_percent_intensity_layer,
                layer_max_avg_pxl_intensity,
                layer_idx_max_avg_pxl_intensity,
                layer_min_avg_pxl_intensity,
                layer_idx_min_avg_pxl_intensity,
                fifty_percent_api_idx,
            ) = compute_layer_intensity_metrics(column_map, nonzero_layers, tiff_base_path, mouse_id)
            df.append([mouse_id, col,
                "N/A", label, num_layers, start_layer, total_volume, fifty_percent_depth,
                max_plane_area, max_area_plane_name, min_plane_area, min_area_plane_name,
                avg_intensity, max_intensity, max_layer, min_intensity, min_layer, fifty_percent_intensity_layer, layer_max_avg_pxl_intensity, start_layer + layer_idx_max_avg_pxl_intensity, layer_min_avg_pxl_intensity, start_layer + layer_idx_min_avg_pxl_intensity, start_layer + fifty_percent_api_idx
                
            ])
        col_df = pd.DataFrame(df, columns=["minian", "Col",
            "minian_id", "column_id", "num_layers", "start_layer", "total_volume", "50%_percent_volume_idx",
            "max_plane_area", "max_plane_area_idx", "min_plane_area", "min_plane_area_idx",
            "avg_intensity", "max_intensity", "max_intensity_idx", "min_intensity", "min_intensity_idx",
            "50%_intensity_idx", "max_API", "max_API_idx", "min_API", "min_API_idx", "50%_API_idx"
        ])
        col_df.to_csv(output_csv_path, index=False)
        print(f"Stats of Column {col} saved to {output_csv_path}")
        all_df.append(col_df)

    final_df = pd.concat(all_df, ignore_index=True)
    final_csv = os.path.join(main_dir, f"{mouse_id}_3D_All_Columns_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    final_df.to_csv(final_csv, index=False)
    print(f"Stats of all columns saved to {final_csv}")


def layer_metric(layer_map, tiff_path):
    """Compute intensity sum and mask area for one layer map and TIFF image."""
    img = tifffile.imread(tiff_path)
    mask = layer_map > 0
    intensity = img[mask].sum()
    area = np.sum(mask)
    return intensity, area


def save_layer_metrics_column(minian_id, session, mouse_id):
    """Save per-layer area and intensity metrics for one Minian ID across days."""

    print(f"Saving layer metrics for Minian {minian_id} and session {session}")
    days = ["D0", "D1", "D3", "D7"]
    base_dir = r"G:\AK\Cell_Overlap\New"

    for day in days:
        print(f"Processing day {day}")
        pattern = os.path.join(
                    base_dir,
                    f"{mouse_id}_3D_{day}",
                    "saved_column_pairs",
                    f"Minian_{session}_Col_{day}",
                    f"minian_{minian_id}_column*.npz"
                )
        
        col_map_paths = glob.glob(pattern)
        if not col_map_paths:
            print(f"No files found for pattern: {pattern}")
            continue
        col_map_path = col_map_paths[0]
        print(f"Loading column map from {col_map_path}")
    
        col_map_npz = np.load(col_map_path)

        tiff_base_path = os.path.join(base_dir, f"{mouse_id}_3D_{day}", "PRE_PROCESSED_TIFF")
        column_map = col_map_npz['column_map']

        non_zero_layers = np.where(np.any(column_map > 0, axis=(1, 2)))[0]
        intensities = []
        areas = []
        layer_names = []
        for layer in non_zero_layers:
            layer_file_name_tif = f"{mouse_id}_P{layer+1:01d}.tif"
            tiff_path = os.path.join(tiff_base_path, layer_file_name_tif)
            intensity, area = layer_metric(column_map[layer], tiff_path)
            intensities.append(intensity)
            areas.append(area)
            layer_names.append(f"P{layer+1}")

        df = pd.DataFrame({
            "Layer": layer_names,
            "Intensity": intensities,
            "Area": areas
        })
        layer_csv = os.path.join(base_dir, f"{mouse_id}_minian_{session}-{minian_id}_{day}_layers.csv")
        df.to_csv(layer_csv, index=False)
        print(f"Layer metrics saved to {layer_csv} for day {day}")


def generate_and_save_stats(saved_pairs_dir, tiff_base_path, output_csv_path):
    """Generate CSV metrics for all .npz pair files in *saved_pairs_dir*.

    This is called by the GUI's "Generate Stats" button.

    Args:
        saved_pairs_dir: Directory containing .npz pair files.
        tiff_base_path: Directory containing per-plane TIFF files for intensity metrics.
        output_csv_path: Destination CSV file path for the results.
    """
    mouse_id = os.path.basename(os.path.dirname(saved_pairs_dir)).split("_")[0]
    npz_files = glob.glob(os.path.join(saved_pairs_dir, "**", "*.npz"), recursive=True)
    npz_files += glob.glob(os.path.join(saved_pairs_dir, "*.npz"))
    npz_files = list(set(npz_files))

    if not npz_files:
        print(f"No .npz files found in: {saved_pairs_dir}")
        return

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    rows = []
    header = [
        "minian_id", "column_id", "num_layers", "start_layer", "total_volume",
        "50%_volume_idx", "max_plane_area", "max_plane_area_name",
        "min_plane_area", "min_plane_area_name",
        "avg_intensity", "max_intensity", "max_intensity_layer",
        "min_intensity", "min_intensity_layer", "50%_intensity_layer",
        "max_API", "max_API_layer_idx", "min_API", "min_API_layer_idx", "50%_API_idx",
    ]

    for pair_path in npz_files:
        data = load_saved_pair(pair_path)
        column_map = data["column_map"]
        minian_id = data["minian_id"]
        column_id = data["column_id"]

        (
            num_layers, start_layer, total_volume, fifty_pct_depth,
            nonzero_layers, max_plane_area, max_area_name,
            min_plane_area, min_area_name,
        ) = compute_column_metrics(column_map)

        (
            avg_intensity, max_intensity, max_layer,
            min_intensity, min_layer, fifty_pct_intensity_layer,
            max_api, max_api_idx, min_api, min_api_idx, fifty_pct_api_idx,
        ) = compute_layer_intensity_metrics(column_map, nonzero_layers, tiff_base_path, mouse_id)

        rows.append([
            minian_id, column_id, num_layers, start_layer, total_volume,
            fifty_pct_depth, max_plane_area, max_area_name,
            min_plane_area, min_area_name,
            avg_intensity, max_intensity, max_layer,
            min_intensity, min_layer, fifty_pct_intensity_layer,
            max_api,
            (start_layer + max_api_idx) if start_layer is not None and max_api_idx is not None else None,
            min_api,
            (start_layer + min_api_idx) if start_layer is not None and min_api_idx is not None else None,
            (start_layer + fifty_pct_api_idx) if start_layer is not None and fifty_pct_api_idx is not None else None,
        ])

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Stats saved to {output_csv_path}")


if __name__ == "__main__":
    """Run batch stats generation for predefined mice."""
    all_column_stats("SG006")
    all_column_stats("SG008")
    # save_layer_metrics_column(100, "D3", "SG006")


 
