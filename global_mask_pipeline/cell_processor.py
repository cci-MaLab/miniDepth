# cell_processor.py
import os
import numpy as np
import tifffile
import cv2
import csv
import pickle
from scipy.spatial import cKDTree
import json
from pathlib import Path

def save_mask(cell_id, mask, directory, mask_format="npy"):
    """
    Saves the binary mask for a cell to a file.
    
    :param cell_id: Identifier for the cell.
    :param mask: Binary mask (numpy array).
    :param directory: Directory to save the mask file.
    :param mask_format: Format to save the mask ("npy", "npz", or "json").
    """
    if mask_format == "npy":
        mask_filename = os.path.join(directory, f"cell_{cell_id}.npy")
        np.save(mask_filename, mask)
        #print(f"Saved mask to {mask_filename}")
    elif mask_format == "npz":
        mask_filename = os.path.join(directory, f"cell_{cell_id}.npz")
        np.savez_compressed(mask_filename, mask=mask)
        print(f"Saved mask to {mask_filename}")
    elif mask_format == "json":
        mask_filename = os.path.join(directory, f"cell_{cell_id}.json")
        with open(mask_filename, "w") as f:
            json.dump(mask.tolist(), f, indent=4)
        print(f"Saved mask to {mask_filename}")
    else:
        print(f"Unsupported mask format: {mask_format}")

def write_centroids_csv(centroid_list, directory, csv_filename="centroids.csv"):
    """
    Writes the centroid information to a CSV file.
    
    :param centroid_list: List of tuples (cell_id, centroid_x, centroid_y).
    :param directory: Directory to save the CSV file.
    :param csv_filename: Name of the CSV file.
    """
    csv_path = os.path.join(directory, csv_filename)
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cell_id", "centroid_x", "centroid_y"])
        for cell_id, cX, cY in centroid_list:
            writer.writerow([int(cell_id), cX, cY])
    print(f"Centroids saved to {csv_path}")

def process_cell(cell_mask):
    """
    Processes a cell mask to compute its centroid.
    
    :param cell_mask: Binary mask (numpy array).
    :return: A tuple (centroid_x, centroid_y).
    """
    
    # Ensure mask is in a numeric type supported by OpenCV moments (uint8).
    # cv2.moments does not accept boolean arrays, which caused the error.
    if cell_mask.dtype == np.bool_:
        mask_for_moments = cell_mask.astype(np.uint8)
    else:
        # For other dtypes, convert to uint8 (non-zero -> 1)
        mask_for_moments = (cell_mask != 0).astype(np.uint8)

    # Compute centroid using image moments.
    M = cv2.moments(mask_for_moments)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = 0, 0
    
    return (cX, cY)




def build_and_save_kd_tree(centroid_list, output_directory, kd_tree_filename="kd_tree.pkl"):
    """
    Builds a k-d tree from the centroid list and saves it as a pickle file.
    
    The centroid_list should be a list of tuples: (cell_id, centroid_x, centroid_y).
    The function builds a cKDTree from the (x, y) coordinates and stores a tuple
    (cell_ids, tree) so that you can later determine which cell corresponds to a given point.
    
    :param centroid_list: List of tuples (cell_id, centroid_x, centroid_y).
    :param output_directory: Directory to save the k-d tree file.
    :param kd_tree_filename: Name of the pickle file.
    """
    if not centroid_list:
        print("No centroids available to build kd-tree.")
        return
    
    cell_ids = [item[0] for item in centroid_list]
    coords = [(item[1], item[2]) for item in centroid_list]
    tree = cKDTree(coords)
    kd_tree_file = os.path.join(output_directory, kd_tree_filename)
    with open(kd_tree_file, "wb") as f:
        pickle.dump((cell_ids, tree), f)
    print(f"KD-tree saved to {kd_tree_file}")

def process_tiff_directory(plane_masks_dir, mask_format="npy", build_kd_tree_flag=False):
    """
    Processes all TIFF files (labelled plane masks) in the given input_directory sequentially.

    For each TIFF file, it computes the binary mask and centroid for each cell.
    The centroids for all cells are written to a CSV file and each mask is saved separately.

    :param plane_masks_dir: Directory containing the TIFF files.
    :param mask_format: Format to save the masks ("npy", "npz", or "json").
    """
    
    plane_masks_dir = Path(plane_masks_dir)

    labelled_masks_planes = list(plane_masks_dir.glob("*.tif")) + \
                            list(plane_masks_dir.glob("*.tiff"))
    
    for plane_mask_filename in labelled_masks_planes:
        plane_name = plane_mask_filename.stem.split('_')[-1]
        plane_mask = tifffile.imread(plane_mask_filename)

        print(f"Loaded plane mask: {plane_name} from {plane_mask_filename}")
        plane_labels = np.unique(plane_mask)
        plane_labels = plane_labels[plane_labels != 0]  # Exclude background label

        print("Processing plane:", plane_name, "with", len(plane_labels), "cells.")

        output_dir = os.path.join(plane_masks_dir, "PROCESSED", f"{plane_name}")
        os.makedirs(output_dir, exist_ok=True)

        centroids = []
        for label in plane_labels:
            mask = (plane_mask == label)
            save_mask(label, mask, output_dir, mask_format=mask_format)

            centroid = process_cell(mask)
            centroids.append((label, centroid[0], centroid[1]))

        print(f"[{plane_name}] Processed {len(plane_labels)} cells. Saving centroids to CSV...")
        write_centroids_csv(centroids, output_dir, csv_filename=f"{plane_name}_centroids.csv")

        if build_kd_tree_flag:
            print(f"[{plane_name}] Building k-d tree from centroids...")
            build_and_save_kd_tree(centroids, output_dir, kd_tree_filename=f"{plane_name}_kd_tree.pkl")
            print(f"[{plane_name}] K-d tree saved.")
        else:
            print(f"[{plane_name}] Skipping k-d tree construction.")
        
        print(f"[{plane_name}] Done.")
    


        
        

