import os
import re
import numpy as np
try:
    from .load import load_preprocessed_layer, get_candidate_cells, load_mask_for_cell
except ImportError:
    # Allow direct script execution from this folder.
    from load import load_preprocessed_layer, get_candidate_cells, load_mask_for_cell


def get_layer_directories(base_dir):
    """
    Returns a list of layer directories under the base directory.
    Assumes that layer directories start with 'L'.
    """
    all_items = os.listdir(base_dir)
    layers = [os.path.join(base_dir, d) for d in all_items if d.startswith('P') and os.path.isdir(os.path.join(base_dir, d))]
    return layers

def compute_overlap(mask1, mask2):
    """
    Calculates the overlap between two binary masks.
    Returns the overlap percentage relative to each mask.
    """
    intersection = np.logical_and(mask1, mask2)
    overlap_area = np.sum(intersection)
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    overlap_percent1 = overlap_area / area1 if area1 > 0 else 0
    overlap_percent2 = overlap_area / area2 if area2 > 0 else 0
    return overlap_percent1, overlap_percent2

def is_match(overlap_percent1, overlap_percent2, threshold=0.6):
    """
    Determines if two cells are a match based on directional overlaps.
    For example, we require that either overlap_percent1 or overlap_percent2 is above threshold.
    """
    return (overlap_percent1 >= threshold) or (overlap_percent2 >= threshold)

def check_match(layer1, layer2, cell_id1, cell_id2, threshold=0.6):
    """
    Check if two cells are a match based on their IDs and overlap.
    """
    #print(f"Checking match between cell {cell_id1} in layer {layer1} and cell {cell_id2} in layer {layer2}")
    mask1 = load_mask_for_cell(layer1, cell_id1)
    mask2 = load_mask_for_cell(layer2, cell_id2)
    overlap_percent1, overlap_percent2 = compute_overlap(mask1, mask2)

    #print(f"Overlap percentages for cell {cell_id1} in layer {layer1} and cell {cell_id2} in layer {layer2}: {overlap_percent1}, {overlap_percent2}")
    return is_match(overlap_percent1, overlap_percent2, threshold)  # Return True if they match


def find_match(curr_cell_id, curr_centroid, curr_layer, next_layer):
    next_layer_data = load_preprocessed_layer(next_layer)
    next_layer_centroids = next_layer_data['centroids']
    next_layer_kdtree = next_layer_data['kd_tree']

    cands = get_candidate_cells(curr_centroid, next_layer_kdtree, 20)

    print(f"{cands} for cell {curr_cell_id} in layer {curr_layer} looking in layer {next_layer}")

    for cand in cands:
        if check_match(curr_layer, next_layer, curr_cell_id, cand):
            return cand
        
    return None


def build_chain(curr_cell_id, curr_centroid, layer_order, cells_visited, curr_layer=None):
    """
    Builds a chain of cell IDs by traversing the layers in the given order.
    Returns a dictionary mapping layer names to cell IDs.

    :param curr_cell_id: The starting cell ID.
    :param curr_centroid: The starting centroid.
    :param layer_order: A dictionary mapping layer names to layer directories.
    :param cells_visited: A dictionary mapping layer names to lists of cell IDs that have been visited.
    :param curr_layer: The starting layer name (optional).
    :return: A dictionary mapping layer names to cell IDs, and a dictionary mapping layer names to lists of cell IDs that have been visited.
    """
    chain = {}
    missed = False  # Flag to indicate if a match is missed
    curr_layer_idx = list(layer_order.keys()).index(curr_layer) if curr_layer else 0  # Get the index of the current layer

    while(not missed and curr_layer_idx < len(layer_order)-1):
        print(f"Building chain at layer {list(layer_order.keys())[curr_layer_idx]} for cell {curr_cell_id}")
        
        next_layer_idx = curr_layer_idx + 1
        next_layer = list(layer_order.keys())[next_layer_idx] if next_layer_idx < len(layer_order) else None
        next_layer_data = load_preprocessed_layer(layer_order[next_layer])

        print(f"Finding match for cell {curr_cell_id} in layer {list(layer_order.keys())[curr_layer_idx]} to layer {next_layer}")

        match_id = find_match(curr_cell_id, curr_centroid, layer_order[curr_layer], layer_order[next_layer])
        if match_id and match_id not in cells_visited[next_layer]:
            cells_visited[curr_layer].append(curr_cell_id)
            cells_visited[next_layer].append(match_id)

            chain[curr_layer] = curr_cell_id
            chain[next_layer] = match_id

            curr_layer_idx = next_layer_idx
            curr_layer = next_layer
            curr_centroid = next_layer_data['centroids'][str(match_id)]
            curr_cell_id = match_id
        else:
            missed = True

    return chain, cells_visited


def compute_chains(base_dir):
    plane_dirs = get_layer_directories(base_dir)

    plane_dirs = sorted(plane_dirs, key= lambda x: int(re.search(r'P(\d+)', x).group(1))) 

    cells_visited = {}
    plane_order = {}
    for i in range(len(plane_dirs)):
        plane_name = plane_dirs[i].split(os.sep)[-1]
        cells_visited[plane_name] = []
        plane_order[plane_name] = plane_dirs[i]

    
    chains = []
    for plane_name, plane_path in plane_order.items():
        print(f"Processing layer: {plane_name} at {plane_path}")

        plane_data = load_preprocessed_layer(plane_path)
        centroids = plane_data['centroids']
        kdtree = plane_data['kd_tree']

        for cell_id, centroid in centroids.items():
            if cell_id not in cells_visited[plane_name]:
                chain, cells_visited = build_chain(cell_id, centroid, plane_order, cells_visited, plane_name)
                if len(chain) > 1:
                    chains.append(chain)

    return chains
