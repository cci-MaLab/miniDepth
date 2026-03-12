"""Batch preprocessing pipeline for static single-photon TIFF stacks.

Main stages per file:
1. Load middle N frames.
2. Remove per-frame glow (morphological opening).
3. Denoise (median + gaussian + optional temporal/speckle cleanup).
4. Rigid subpixel motion correction.
5. Project and remove projection-level background.
6. Save intermediate QC outputs and final projection.
"""

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from typing import Tuple, Literal, Optional
from scipy.ndimage import fourier_shift, gaussian_filter, median_filter
from skimage.registration import phase_cross_correlation
from skimage.morphology import disk, opening

try:
    import tifffile as tiff
    _HAS_TIFFFILE = True
except Exception:
    from skimage.io import imread
    _HAS_TIFFFILE = False



# How many frames to keep for processing
PIPELINE_N_FRAMES = 100  # middle 100

# ---------------- Config ----------------
INPUT_DIR = Path("/N/project/CCK/Inscopix-Second_batch/3D-Project/Chamber_20250806-131756/Raw_Tiff")
OUT_INTERMEDIATE_DIR = Path("/N/project/CCK/Inscopix-Second_batch/3D-Project/Chamber_20250806-131756/Raw_Tiff/intermediate")
OUT_FINAL_DIR = Path("/N/project/CCK/Inscopix-Second_batch/3D-Project/Chamber_20250806-131756/Raw_Tiff/final_proj")
#OUT_INTERMEDIATE_DIR = Path(os.path.join(INPUT_DIR, "intermediate"))
#OUT_FINAL_DIR = Path(os.path.join(INPUT_DIR, "final_proj"))
MIDDLE_N_FRAMES = 100
SOMA_RADIUS_PX = 22
PER_FRAME_OPEN_RADIUS = max(1, int(0.8 * SOMA_RADIUS_PX))  # per-frame glow removal radius
PROJ_OPEN_RADIUS      = max(2, int(2.0 * SOMA_RADIUS_PX))  # projection-level background removal radius
MEDIAN_SIZE = 3
GAUSSIAN_SIGMA = 0.8
UPSAMPLE_FACTOR = 10
TWO_PASS_MC = True

# ------------- Stage folders (flat) -------------
RAW_FRAME1_DIR      = OUT_INTERMEDIATE_DIR / "raw" / "frame1"
RAW_PROJ_DIR        = OUT_INTERMEDIATE_DIR / "raw" / "proj"
GLOW_PROJ_DIR       = OUT_INTERMEDIATE_DIR / "glow_removed" / "proj"
DENOISE_FRAME1_DIR  = OUT_INTERMEDIATE_DIR / "denoising" / "frame1"
DENOISE_PROJ_DIR    = OUT_INTERMEDIATE_DIR / "denoising" / "proj"
MC_PLOTS_DIR        = OUT_INTERMEDIATE_DIR / "motion_correction" / "plots"
MC_BEFORE_PROJ_DIR  = OUT_INTERMEDIATE_DIR / "motion_correction" / "before" / "proj"
MC_AFTER_PROJ_DIR   = OUT_INTERMEDIATE_DIR / "motion_correction" / "after" / "proj"
PROJECTION_PROJ_DIR = OUT_INTERMEDIATE_DIR / "projection" / "proj"
BG_OPEN_DIR         = OUT_INTERMEDIATE_DIR / "background_removal" / "proj" / "bg_opening"
BG_CORR_DIR         = OUT_INTERMEDIATE_DIR / "background_removal" / "proj" / "corrected"


def list_final_proj_files():
    """Return already-produced final projection stems from OUT_FINAL_DIR.

    Used to skip files already processed by this pipeline.
    """
    folder = OUT_FINAL_DIR

    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}")
        return []

    if not folder.is_dir():
        print(f"[ERROR] Path is not a directory: {folder}")
        return []

    print(f"[INFO] Reading files from: {folder}")
    
    result = []
    for file in folder.iterdir():
        if file.is_file():
            ext = file.suffix.lower().strip()
            if ext in [".tiff", ".tif"]:
                print(f"[FOUND] {file.name} → {file.stem}")
                result.append(file.stem)

    print(f"[DONE] Found {len(result)} .tiff/.tif files.")
    return result

# Snapshot existing final outputs once so repeated batch runs can skip them.
PROCESSED_FILES = list_final_proj_files()

# ------------------------------
# Glow removal modules
# ------------------------------

def remove_glow_opening(stack: np.ndarray, radius: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Morphological opening to estimate smooth background ("glow") per frame, then subtract.
    radius: structuring element radius; choose larger than background features, smaller than soma size * 1.5–2x.
    Returns (bg, corrected_stack).
    """
    print(f"[STEP] Glow removal via morphological opening (radius={radius})...")
    selem = disk(radius)
    T = stack.shape[0]
    bg = np.empty_like(stack)
    corrected = np.empty_like(stack)
    for t in range(T):
        b = opening(stack[t], selem)
        bg[t] = b
        corrected[t] = stack[t] - b
        corrected[t][corrected[t] < 0] = 0
    print("[DONE] Glow removal (opening) complete.")
    return bg, corrected


# ------------------------------
# Denoising Modules
# ------------------------------

def denoise_median(stack: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Apply a per-frame median filter. Good for removing hot/cold pixels.
    """

    print(f"[STEP] Median denoising with size={size}...")
    T = stack.shape[0]
    out = np.empty_like(stack)
    for t in range(T):
        out[t] = median_filter(stack[t], size=size)
    print("[DONE] Median denoising complete.")
    return out


def denoise_gaussian(stack: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    Apply a per-frame Gaussian filter. Good for mild shot/read noise.
    Keep sigma small to preserve soma edges.
    """
    print(f"[STEP] Gaussian denoising with sigma={sigma}...")
    T = stack.shape[0]
    out = np.empty_like(stack)
    for t in range(T):
        out[t] = gaussian_filter(stack[t], sigma=sigma)
    print("[DONE] Gaussian denoising complete.")
    return out

def temporal_median(stack, win=3):
    """Apply temporal median filtering with window `win` along the frame axis."""
    return median_filter(stack, size=(win,1,1))

def small_speckle_opening(stack, radius=2):
    """Apply small-radius opening per frame to remove isolated speckle artifacts."""
    se = disk(radius)
    out = stack.copy()
    for t in range(stack.shape[0]):
        out[t] = opening(out[t], se)
    return out

# ------------------------------
# Motion Correction Modules
# ------------------------------


def build_template(
    stack: np.ndarray,
    method: Literal["median", "mean"] = "median",
    use_middle_n: Optional[int] = 50
) -> np.ndarray:
    """
    Build a motion-correction template from the middle n frames.
    """
    T = stack.shape[0]
    if use_middle_n is None or use_middle_n > T:
        use_middle_n = T
    start = (T - use_middle_n) // 2
    end = start + use_middle_n
    sub = stack[start:end]
    print(f"[STEP] Building template from frames [{start}:{end}) using {method}.")
    if method == "median":
        return np.median(sub, axis=0).astype(np.float32)
    else:
        return sub.mean(axis=0).astype(np.float32)


def rigid_motion_correction(
    stack: np.ndarray,
    ref: Optional[np.ndarray] = None,
    upsample_factor: int = 10,
    two_pass: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rigid subpixel motion correction:
      - If ref is None, compute a template (median of middle frames).
      - Estimate shift per frame via phase_cross_correlation.
      - Apply shift via Fourier shift.
      - Optionally do a second pass using the corrected average as new template.
    Returns:
      shifts: (T, 2) array of (dy, dx)
      corrected_stack: same shape as input
    """
    print("[STEP] Rigid motion correction (phase correlation)...")
    T = stack.shape[0]
    if ref is None:
        ref = build_template(stack, method="median", use_middle_n=min(T, 50))

    shifts = np.zeros((T, 2), dtype=np.float32)
    corrected = np.empty_like(stack)

    for t in range(T):
        shift, error, _ = phase_cross_correlation(ref, stack[t], upsample_factor=upsample_factor)
        shifts[t] = shift
        F = np.fft.fftn(stack[t])
        corrected[t] = np.fft.ifftn(fourier_shift(F, shift=shift)).real.astype(np.float32)

    if two_pass:
        print("[INFO] Second pass refinement using corrected average as template.")
        ref2 = corrected.mean(axis=0).astype(np.float32)
        shifts2 = np.zeros_like(shifts)
        corrected2 = np.empty_like(stack)
        for t in range(T):
            shift, error, _ = phase_cross_correlation(ref2, corrected[t], upsample_factor=upsample_factor)
            shifts2[t] = shift
            F = np.fft.fftn(corrected[t])
            corrected2[t] = np.fft.ifftn(fourier_shift(F, shift=shift)).real.astype(np.float32)
        shifts = shifts + shifts2
        corrected = corrected2

    print("[DONE] Motion correction complete.")
    return shifts, corrected


def plot_shifts(shifts: np.ndarray, title: str = "Estimated shifts (dy, dx)"):
    """Display dy/dx frame-wise rigid shifts for quick motion QC."""
    plt.figure()
    plt.plot(shifts[:,0], label="dy")
    plt.plot(shifts[:,1], label="dx")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Shift (pixels)")
    plt.legend()
    plt.show()

def frame_template_correlation(stack: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Compute normalized correlation per frame with template as a QC metric.
    """
    t_flat = template.ravel()
    t_mu = t_flat.mean()
    t_std = t_flat.std() + 1e-8
    corrs = np.zeros(stack.shape[0], dtype=np.float32)
    for i in range(stack.shape[0]):
        f = stack[i].ravel()
        f_mu = f.mean()
        f_std = f.std() + 1e-8
        corrs[i] = np.dot((f - f_mu), (t_flat - t_mu)) / (len(f) * t_std * f_std)
    return corrs

def plot_correlations(corrs_before: np.ndarray, corrs_after: np.ndarray):
    """Display frame-template correlation before/after motion correction."""
    plt.figure()
    plt.plot(corrs_before, label="Before MC")
    plt.plot(corrs_after, label="After MC")
    plt.title("Frame-to-template correlation (Before vs After)")
    plt.xlabel("Frame")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()


# --------------------------------------------
# Background Subtraction & Projection Modules
# --------------------------------------------

def select_middle_frames(stack: np.ndarray, n: int = 100) -> np.ndarray:
    """
    Select the middle n frames from a stack of shape (T, H, W).
    """
    T = stack.shape[0]
    if n > T:
        raise ValueError(f"Requested {n} frames but stack only has {T}.")
    start = (T - n) // 2
    end = start + n
    print(f"[STEP] Selecting middle {n} frames: indices [{start}:{end}) out of T={T}")
    return stack[start:end]


def project_stack(stack: np.ndarray, method: Literal["mean","median","trimmed_mean"]="mean", trimmed_frac: float=0.1) -> np.ndarray:
    """
    Temporal projection to a single 2D image.
    """
    print(f"[STEP] Projection | method={method}")
    T = stack.shape[0]
    if method == "mean":
        proj = stack.mean(axis=0)
    elif method == "median":
        proj = np.median(stack, axis=0)
    elif method == "trimmed_mean":
        k = int(trimmed_frac * T)
        sortd = np.sort(stack, axis=0)
        proj = sortd[k:T-k].mean(axis=0) if T > 2*k else sortd.mean(axis=0)
    else:
        raise ValueError("Unknown projection method.")
    print("[DONE] Projection.")
    return proj.astype(np.float32)


def bg_remove_projection_opening(proj: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projection-level morphological background removal (white top-hat):
    bg = opening(proj, selem); corrected = proj - bg (clamped at 0).
    """
    print(f"[STEP] Projection background via opening | radius={radius}")
    se = disk(max(1, int(radius)))
    bg = opening(proj, se)
    corrected = proj - bg
    corrected[corrected < 0] = 0
    print("[DONE] Projection background removal.")
    return bg.astype(np.float32), corrected.astype(np.float32) 


# ------------------------------
# Visualization Helpers
# ------------------------------

def show_image(img: np.ndarray, title: str = "", cmap: str = "gray"):
    """
    Display a single image with Matplotlib.
    Requirement: Each chart should be its own plot; do not set specific colors/styles.
    """
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def minmax_norm(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] for fair visualization."""
    mn, mx = float(img.min()), float(img.max())
    return (img - mn) / (mx - mn + 1e-8)




# --------- Small I/O helpers ----------
def ensure_dir(p: Path):
    """Create directory `p` (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

def save_png(image: np.ndarray, path: Path, title: str = None):
    """Save a normalized preview PNG for visual QC."""
    ensure_dir(path.parent)
    # Normalize preview only; this does not affect analysis data.
    img = minmax_norm(image)
    plt.figure()
    plt.imshow(img, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_tiff(image: np.ndarray, path: Path):
    """Save TIFF output; float images are scaled to uint16 for consistency."""
    ensure_dir(path.parent)
    if image.dtype.kind == "f":
        img = minmax_norm(image)
        img = (img * 65535).astype(np.uint16)
    else:
        img = image
    imwrite(str(path), img)

def plot_and_save_shifts(shifts: np.ndarray, path: Path, title: str = "Estimated subpixel shifts (dy, dx)"):
    """Save dy/dx shift plot to disk."""
    ensure_dir(path.parent)
    plt.figure()
    plt.plot(shifts[:,0], label="dy")
    plt.plot(shifts[:,1], label="dx")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Shift (pixels)")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_and_save_correlations(corrs_before: np.ndarray, corrs_after: np.ndarray, path: Path):
    """Save before/after correlation QC plot to disk."""
    ensure_dir(path.parent)
    plt.figure()
    plt.plot(corrs_before, label="Before MC")
    plt.plot(corrs_after, label="After MC")
    plt.title("Frame-to-template correlation (Before vs After)")
    plt.xlabel("Frame")
    plt.ylabel("Correlation")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# ------------- Pipeline per file -------------

def process_one_file(tif_path: Path):
    """Run full preprocessing pipeline on one TIFF stack.

    Args:
        tif_path: Path to input stack file.
    """
    base = tif_path.stem  # e.g., SG008_P1
    print(f"\n[FILE] Processing {base}")

    if base in PROCESSED_FILES:
        print(f"[SKIP] Already processed {base}.")
        return
    
    # ---------- LOAD (middle N frames only) ----------
    if _HAS_TIFFFILE:
        with tiff.TiffFile(str(tif_path)) as tf:
            n_frames = len(tf.pages)
            count = min(PIPELINE_N_FRAMES, n_frames)
            start = max(0, (n_frames - count) // 2)
            # Read only needed pages to reduce memory use for long stacks.
            stack = tf.series[0].asarray(key=slice(start, start + count)).astype(np.float32)
    else:
        # Fallback path when tifffile API is unavailable.
        full = imread(str(tif_path)).astype(np.float32)
        n_frames = full.shape[0]
        count = min(PIPELINE_N_FRAMES, n_frames)
        start = max(0, (n_frames - count) // 2)
        stack = full[start:start + count]
        del full  # free memory

    print(f"[LOAD] {base}: shape={stack.shape}, dtype={stack.dtype} (using middle {stack.shape[0]} frames)")

    # ---------- RAW VISUALS (first kept frame + mean projection) ----------
    save_png(stack[0], RAW_FRAME1_DIR / f"{base}.png", title=f"{base} frame 1 (normalized)")
    raw_proj = project_stack(stack, method="mean")
    save_tiff(raw_proj, RAW_PROJ_DIR / f"{base}.tif")

    # ---------- Per-frame glow removal ----------
    _, corrected = remove_glow_opening(stack, radius=PER_FRAME_OPEN_RADIUS)
    glow_mean_proj = project_stack(corrected, method="mean")
    save_tiff(glow_mean_proj, GLOW_PROJ_DIR / f"{base}.tif")

    # ---------- Denoising stack ----------
    med = denoise_median(corrected, size=MEDIAN_SIZE)
    med_gauss = denoise_gaussian(med, sigma=GAUSSIAN_SIGMA)
    save_png(med_gauss[0], DENOISE_FRAME1_DIR / f"{base}.png", title=f"{base} denoised frame 1")

    med_temporal = temporal_median(med_gauss, win=5)
    deno_speck = small_speckle_opening(med_temporal, radius=3)
    save_png(deno_speck[0], DENOISE_FRAME1_DIR / f"{base}_denoised_speckle.png", title=f"{base} denoised frame 1 (speckle removed)")
    denoise_proj = project_stack(deno_speck, method="mean")
    save_tiff(denoise_proj, DENOISE_PROJ_DIR / f"{base}.tif")

    # ---------- Motion correction and QC metrics ----------
    middle_for_template = select_middle_frames(deno_speck, n=min(MIDDLE_N_FRAMES, deno_speck.shape[0]))
    ref = build_template(middle_for_template, use_middle_n=middle_for_template.shape[0])
    corrs_before = frame_template_correlation(deno_speck, ref)
    shifts, mc_stack = rigid_motion_correction(
        deno_speck, ref=ref, upsample_factor=UPSAMPLE_FACTOR, two_pass=TWO_PASS_MC
    )
    corrs_after = frame_template_correlation(mc_stack, ref)

    plot_and_save_shifts(shifts, MC_PLOTS_DIR / f"{base}_shifts.png")
    plot_and_save_correlations(corrs_before, corrs_after, MC_PLOTS_DIR / f"{base}_correlations.png")
    save_tiff(denoise_proj, MC_BEFORE_PROJ_DIR / f"{base}.tif")
    after_mc_proj_all = project_stack(mc_stack, method="mean")
    save_tiff(after_mc_proj_all, MC_AFTER_PROJ_DIR / f"{base}.tif")

    # ---------- Final projection over middle N corrected frames ----------
    middle = select_middle_frames(mc_stack, n=min(MIDDLE_N_FRAMES, mc_stack.shape[0]))
    proj = project_stack(middle, method="mean")
    save_tiff(proj, PROJECTION_PROJ_DIR / f"{base}.tif")

    # ---------- Projection-level background removal (white top-hat style) ----------
    bg2, proj_corr = bg_remove_projection_opening(proj, radius=PROJ_OPEN_RADIUS)
    save_tiff(bg2,       BG_OPEN_DIR / f"{base}.tif")
    save_tiff(proj_corr, BG_CORR_DIR / f"{base}.tif")

    # ---------- Final output ----------
    save_tiff(proj_corr, OUT_FINAL_DIR / f"{base}.tif")
    print(f"[DONE] Saved final projection → {OUT_FINAL_DIR / f'{base}.tif'}")

# ------------- Batch runner -------------
def process_all_images():
    """Process all matching SG006 input stacks found in INPUT_DIR."""
    # Ensure base dirs
    for d in [
        OUT_FINAL_DIR, OUT_INTERMEDIATE_DIR,
        RAW_FRAME1_DIR, RAW_PROJ_DIR,
        GLOW_PROJ_DIR,
        DENOISE_FRAME1_DIR, DENOISE_PROJ_DIR,
        MC_PLOTS_DIR, MC_BEFORE_PROJ_DIR, MC_AFTER_PROJ_DIR,
        PROJECTION_PROJ_DIR,
        BG_OPEN_DIR, BG_CORR_DIR,
    ]:
        ensure_dir(d)

    # Match SG006_P*.tif / .tiff inside INPUT_DIR
    candidates = sorted(list(INPUT_DIR.glob("SG006_P*.tif")) + list(INPUT_DIR.glob("SG006_P*.tiff")))
    if not candidates:
        print(f"[WARN] No input files found in {INPUT_DIR}. Expected SG006_P*.tif/.tiff")
        return

    print(f"[INFO] Found {len(candidates)} files.")
    for tif_path in candidates:
        try:
            process_one_file(tif_path)
        except Exception as e:
            print(f"[ERROR] Failed on {tif_path.name}: {e}")

if __name__ == "__main__":
    process_all_images()
    print("[INFO] Done.")