"""Core data-loading, overlap-analysis, and 3D visualization helpers.

This module contains shared logic used by both the interactive Napari viewer
and the offline stats scripts.
"""

import os
from os import listdir
from os.path import isdir, isfile
from os.path import join as pjoin

import dask.array as darr
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import xarray as xr
from scipy.ndimage import binary_dilation

def load_volume(output_dir, num_layers):
    """Load per-plane global label arrays and stack them into a 3D volume.

    The loader expects files named like `plane_001_global.npy` inside
    `output_dir`. Missing plane files are skipped with a warning.

    Args:
        output_dir: Directory containing per-plane global label files.
        num_layers: Number of expected planes (1-indexed).

    Returns:
        A NumPy array with shape `(z, y, x)`.

    Raises:
        ValueError: If no layers could be loaded.
    """
    stack = []
    print(f"Loading volume from {output_dir} with {num_layers} layers.")
    for layer_idx in range(1, num_layers + 1):
        file_path = os.path.join(output_dir, f"plane_{layer_idx:03d}_global.npy")
        if os.path.exists(file_path):
            global_label = np.load(file_path)
            stack.append(global_label)
        else:
            print(f"Warning: {file_path} does not exist.")
    if not stack:
        raise ValueError("No layers loaded. Check your npy files and output directory.")
    volume = np.stack(stack, axis=0)
    working_dir = os.path.join(os.curdir, "cache")
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    np.save(os.path.join(working_dir, "volume.npy"), volume)
    return volume


def near_by_cells(main_cells, volume, dilation_size=1, overlap_threshold=0.2):
    """Find overlapping and adjacent labels for one or more selected cells.

    Overlap is estimated in 2D projection space (Y-X) by collapsing across Z,
    then comparing overlap ratios. Additional adjacency is detected with 3D
    morphological dilation of the selected cells.

    Args:
        main_cells: Iterable of label IDs to analyze.
        volume: Full labeled volume `(z, y, x)`.
        dilation_size: Non-zero value used in dilation structuring mask.
        overlap_threshold: Ratio threshold used to classify overlap.

    Returns:
        Tuple `(overlapping_cells, adjacent_cells)` where both values are
        sorted lists of label IDs.
    """
    overlapping_cells = set()
    adjacent_cells = set()

    for main_cell in main_cells:
        main_cell_mask = (volume == main_cell)
        main_cell_2d_mask = np.any(main_cell_mask, axis=0)
        selected_cells = volume[:, main_cell_2d_mask]
        unique_cells = np.unique(selected_cells[selected_cells != 0])

        for cell in unique_cells:
            if cell == main_cell:
                continue
            cell_mask = (volume == cell)
            cell_2d_mask = np.any(cell_mask, axis=0)
            overlap_mask = cell_2d_mask & main_cell_2d_mask
            overlap_area = np.sum(overlap_mask)
            cell_area = np.sum(cell_2d_mask)
            main_area = np.sum(main_cell_2d_mask)

            if cell_area > 0 and overlap_area / cell_area >= overlap_threshold:
                overlapping_cells.add(cell)
            elif main_area > 0 and overlap_area / main_area >= overlap_threshold:
                overlapping_cells.add(cell)
            else:
                adjacent_cells.add(cell)

        # Expand the selected cell in 3D to capture spatial neighbors.
        dilated = binary_dilation(
            main_cell_mask,
            structure=np.ones((5, 5, 5)) * dilation_size,
        )
        dilated_cells = volume[dilated]
        adjacent_cells.update(
            dilated_cells[(dilated_cells != 0) & (dilated_cells != main_cell) & (~np.isin(dilated_cells, list(overlapping_cells)))]
        )

    return sorted(overlapping_cells), sorted(adjacent_cells)


def open_minian(dpath: str, return_dict=True):
    """Open MINIAN data from a file/folder into xarray structures.

    Args:
        dpath: Path to a MINIAN dataset file or directory.
        return_dict: If True and `dpath` is a directory, return a dictionary
            of DataArrays keyed by array name. If False, merge into one dataset.

    Returns:
        A lazily loaded xarray object (dataset or dict of arrays).

    Raises:
        FileNotFoundError: If `dpath` is neither a file nor a directory.
    """
    if isfile(dpath):
        # Single MINIAN file path.
        ds = xr.open_dataset(dpath).chunk()
    elif isdir(dpath):
        # Multi-array MINIAN directory.
        dslist = []
        for d in listdir(dpath):
            arr_path = pjoin(dpath, d)
            if isdir(arr_path):
                # Each subdirectory is a Zarr array.
                arr = list(xr.open_zarr(arr_path).values())[0]
                arr.data = darr.from_zarr(
                    os.path.join(arr_path, arr.name), inline_array=True
                )
                dslist.append(arr)
        if return_dict:
            ds = {d.name: d for d in dslist}
        else:
            ds = xr.merge(dslist, compat="no_conflicts")
    else:
        raise FileNotFoundError(f"{dpath} is not a file or directory")
    return ds


def find_overlapping_cells(minian_data, volume):
    """Map each MINIAN unit to overlapping column labels in the 3D volume.

    Args:
        minian_data: Data structure containing MINIAN footprints under key `A`.
        volume: Labeled volume `(z, y, x)` for column IDs.

    Returns:
        Dictionary: `{unit_id: [column_label, ...]}`.
    """
    overlap_dict = {}
    for unit_id in minian_data["A"]["unit_id"].values:
        print(f"Processing unit_id: {unit_id}")

        footprint = minian_data["A"].sel(unit_id=unit_id).data
        footprint_mask = (footprint > 0)
        overlapping_labels = np.unique(volume[:, footprint_mask])
        overlapping_labels = overlapping_labels[overlapping_labels != 0]
        overlap_dict[unit_id] = overlapping_labels.tolist()
    
    return overlap_dict

def show_label_stack_pyvista(label_volume, z_spacing=10, vertical_axis='z'):
    """Render a quick 3D point-cloud view of labeled slices using PyVista.

    Args:
        label_volume: 3D label array with shape `(z, y, x)`.
        z_spacing: Distance between planes along the stacking axis.
        vertical_axis: Stacking axis in 3D scene (`'z'` or `'y'`).

    Raises:
        ValueError: If `vertical_axis` is not one of `'z'` or `'y'`.
    """
    plotter = pv.Plotter()
    z_dim, y_dim, x_dim = label_volume.shape

    # Get all unique label values except background (0)
    unique_vals = np.unique(label_volume)
    unique_vals = unique_vals[unique_vals != 0]
    cmap = plt.cm.get_cmap('tab10', len(unique_vals))
    label_to_color = {
        val: tuple((np.array(cmap(i))[:3] * 255).astype(int))
        for i, val in enumerate(unique_vals)
    }

    # Plot all slices
    for z in range(z_dim):
        slice_data = label_volume[z]
        for label_val in unique_vals:
            coords = np.argwhere(slice_data == label_val)
            if coords.size == 0:
                continue

            # Compute 3D coordinates with spacing
            if vertical_axis == 'z':
                points = np.column_stack((coords[:, 1], coords[:, 0], np.full(len(coords), z * z_spacing)))
            elif vertical_axis == 'y':
                points = np.column_stack((coords[:, 1], np.full(len(coords), z * z_spacing), coords[:, 0]))
            else:
                raise ValueError("vertical_axis must be 'z' or 'y'")

            cloud = pv.PolyData(points)
            color = label_to_color[label_val]
            plotter.add_mesh(cloud, color=color, point_size=2, render_points_as_spheres=True)

        # Add a text label between planes to make depth easier to read.
        z_pos = z * z_spacing + z_spacing / 2
        center_x = x_dim // 2
        center_y = y_dim // 2

        label_pos = {
            'z': (center_x, center_y, z_pos),
            'y': (center_x, z_pos, center_y)
        }[vertical_axis]

        plotter.add_point_labels(
            [label_pos],
            [f"Layer {z}"],
            font_size=12,
            point_color='white',
            text_color='black',
            show_points=False,
            always_visible=True
        )

    # Add axes and set default view
    plotter.add_axes(interactive=True)
    if vertical_axis == 'z':
        plotter.view_yz()
    elif vertical_axis == 'y':
        plotter.view_xz()

    plotter.show()