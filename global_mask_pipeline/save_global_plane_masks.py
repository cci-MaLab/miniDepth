import os
import pickle

try:
    from .cell_processor import process_tiff_directory
    from .chains import compute_chains
    from .global_masks import create_global_planes
except ImportError:
    # Allow direct script execution from this folder.
    from cell_processor import process_tiff_directory
    from chains import compute_chains
    from global_masks import create_global_planes


def run_pipeline(tiff_dir, pkl_file, num_planes=180):
    """Run end-to-end global plane mask generation for one labeled-mask directory."""
    root = os.path.join(tiff_dir, "PROCESSED")
    out_dir = os.path.join(tiff_dir, "GLOBAL_PLANE_MASKS")

    process_tiff_directory(tiff_dir, build_kd_tree_flag=True)

    chains = compute_chains(root)
    with open(pkl_file, "wb") as f:
        pickle.dump(chains, f)

    with open(pkl_file, "rb") as f:
        chains = pickle.load(f)

    mapping, collisions = create_global_planes(chains, root, out_dir, num_planes)
    return mapping, collisions


if __name__ == "__main__":
    TIFF_DIR = "/N/slate/akorada/CalDepth/runs_unet/20251010-132615/Within_Animal/SG006_3D_D7/Labeled_Masks"
    PKL_FILE = "SG006_3D_D7_chains.pkl"
    NUM_PLANES = 180
    run_pipeline(TIFF_DIR, PKL_FILE, NUM_PLANES)