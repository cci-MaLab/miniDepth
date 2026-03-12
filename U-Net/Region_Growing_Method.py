"""Region-growing segmentation from local intensity maxima.

Pipeline summary:
1. Detect local maxima as seed centers.
2. For each seed and angle, trace a ray and select a candidate edge pixel
    using the strongest intensity-drop slope.
3. Convert each selected ray to discrete pixels via Bresenham lines.
4. Run constrained region growth from those initialized pixels.
5. Save visualization and mask outputs.
"""

from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import numpy as np
import math
import tifffile as tiff
from collections import deque
import cv2


def read_gray01(path: str, how="Normalised") -> np.ndarray:
    """Load an image as grayscale and optionally normalize it to [0, 1].

    Args:
        path: Input image path.
        how: Use "Normalised" for min-max normalization, any other value to keep
            original image range and dtype.

    Returns:
        2D numpy array image.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"Image not found: {path}"
    if how != "Normalised":
        return img
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn + 1e-8)
    return img


def read_show_local_max(image_path, min_distance=15, threshold_rel=0.35):
    """Detect and visualize local maxima used as seed centers.

    Returns:
        Tuple of (raw grayscale image, local-max coordinate array).
    """
    img = read_gray01(image_path, how="Raw")
    coordinates = peak_local_max(
        img, min_distance=min_distance, threshold_rel=threshold_rel
    )
    plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap="gray")
    plt.scatter(coordinates[:, 1], coordinates[:, 0], color="red", s=10)
    plt.savefig("local_max.png")
    plt.show()
    return img, coordinates


def bresenham(x1, y1, x2, y2):
    """Return discrete line pixels between two points using Bresenham's algorithm."""
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    if dx > dy:
        p = 2 * dy - dx
        while x1 != x2:
            points.append((x1, y1))
            if p >= 0:
                y1 += sy
                p -= 2 * dx
            x1 += sx
            p += 2 * dy
    else:
        p = 2 * dx - dy
        while y1 != y2:
            points.append((x1, y1))
            if p >= 0:
                x1 += sx
                p -= 2 * dy
            y1 += sy
            p += 2 * dx
    points.append((x2, y2))
    return points


def generate_edge_pixels(coords, img, max_radius=20, alpha=0.40, num_angles=16):
    """Generate radial edge hypotheses for each local-maximum seed.

    For each seed and ray angle, the code steps outward up to max_radius.
    Candidate pixels must stay above alpha * center_intensity; among them,
    the pixel with the largest intensity-drop slope is selected as the edge.

    Returns:
        A list of dictionaries. Each dictionary stores one seed, its rays,
        and the union of Bresenham line pixels from selected rays.
    """
    edge_dict_list = []
    theta_values = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    for x0, y0 in coords:
        print(f"Starting ray tracing from center point: ({x0}, {y0})")
        print("-" * 30)
        edge_dict = {"local_maxima": (x0, y0), "rays": []}
        intensity_center = img[x0, y0]
        print(f"    Intensity at center pixel ({x0}, {y0}) is {intensity_center:.4f}")
        for theta in theta_values:
            print(
                f"  -- Current theta: {math.degrees(theta):.2f} degrees "
                f"for local maxima ({x0}, {y0}) --"
            )
            print(f"    Intensity at center pixel ({x0}, {y0}) is {intensity_center:.4f}")
            best_slope = -math.inf
            best_pixel = None
            for r in range(1, max_radius):
                # Parametric ray point at distance r for the current angle.
                x = x0 + r * math.cos(theta)
                y = y0 + r * math.sin(theta)
                x_rounded, y_rounded = int(round(x)), int(round(y))
                if (
                    x_rounded < 0
                    or x_rounded >= img.shape[0]
                    or y_rounded < 0
                    or y_rounded >= img.shape[1]
                ):
                    print(
                        f"    Pixel ({x_rounded}, {y_rounded}) is out of bounds. "
                        "Stopping this ray."
                    )
                    break
                intensity_current = img[x_rounded, y_rounded]

                if intensity_current > alpha * intensity_center:
                    # Maximize center-to-current intensity drop per radius step.
                    slope = (intensity_center - intensity_current) / r
                    print(
                        f"    Intensity at current pixel ({x}, {y}) "
                        f"({x_rounded}, {y_rounded}) is {intensity_current:.4f}, "
                        f"slope is {slope:.4f}"
                    )
                    if slope > best_slope:
                        best_slope = slope
                        best_pixel = (x_rounded, y_rounded)
                        print(
                            f"    New best slope found: {best_slope:.4f} "
                            f"at pixel {best_pixel}"
                        )
                    else:
                        print("    Current slope is less than best slope.")
                else:
                    print(
                        f"    Intensity at current pixel ({x_rounded}, {y_rounded}) is 0."
                    )
                    # Stop scanning this direction once intensity falls below gate.
                    break
            if best_pixel is not None:
                # Rasterize center-to-edge segment so region growing starts from
                # connected support pixels, not sparse endpoints only.
                ray_pixels = bresenham(x0, y0, best_pixel[0], best_pixel[1])
                edge_dict["rays"].append(
                    {
                        "theta": theta,
                        "edge": best_pixel,
                        "pixels": ray_pixels,
                    }
                )
                print(
                    f"  Best edge pixel for theta {math.degrees(theta):.2f} degrees: "
                    f"{best_pixel} with slope {best_slope:.4f}"
                )
            else:
                edge_dict["rays"].append({"theta": theta, "edge": None, "pixels": []})
                print(
                    f"  No valid edge pixel found for theta {math.degrees(theta):.2f} degrees."
                )
        all_pixels = set()
        for ray in edge_dict["rays"]:
            all_pixels.update(ray["pixels"])
        edge_dict["labeled_pixels"] = all_pixels
        edge_dict_list.append(edge_dict)
        print("=" * 50)
    return edge_dict_list


def region_grow_constraint2(edge_dict, img, alpha=0.35, max_radius=20):
    """Grow a region from initialized pixels with distance/intensity constraints.

    Rules:
    - Candidate must be inside max_radius from the local-maximum seed.
    - Candidate intensity must be >= alpha * center_intensity.
    - Growth is 8-connected and uses local monotonicity checks relative to
      current frontier pixel and radial direction.
    """
    lm_row, lm_col = edge_dict["local_maxima"]
    intensity_center = img[lm_row, lm_col]

    labeled = set(edge_dict["labeled_pixels"])
    # BFS queue over currently accepted pixels.
    frontier = deque(labeled)

    while frontier:
        ref_row, ref_col = frontier.popleft()
        intensity_ref = img[ref_row, ref_col]
        d_ref = np.hypot(ref_row - lm_row, ref_col - lm_col)

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                cand_row, cand_col = ref_row + dr, ref_col + dc

                if (
                    cand_row < 0
                    or cand_row >= img.shape[0]
                    or cand_col < 0
                    or cand_col >= img.shape[1]
                ):
                    continue

                if (cand_row, cand_col) in labeled:
                    continue

                intensity_cand = img[cand_row, cand_col]
                d_cand = np.hypot(cand_row - lm_row, cand_col - lm_col)
                delta_d = d_cand - d_ref

                if d_cand > max_radius:
                    continue
                if intensity_cand < alpha * intensity_center:
                    continue

                if delta_d <= np.sqrt(2):
                    # Prefer outward/non-decreasing intensity growth.
                    if intensity_cand >= intensity_ref:
                        labeled.add((cand_row, cand_col))
                        frontier.append((cand_row, cand_col))
                    else:
                        # Allow a decrease only when moving back toward seed.
                        if d_cand < d_ref:
                            labeled.add((cand_row, cand_col))
                            frontier.append((cand_row, cand_col))

    return labeled


if __name__ == "__main__":
    # 1) Seed detection
    img, coords = read_show_local_max(
        "/SG013_P5.tif",
        threshold_rel=0.30,
    )

    # 2) Radial edge proposal and line initialization
    edge_dict_list = generate_edge_pixels(coords, img, max_radius=20, num_angles=16)

    # 3) Region growing per seed
    regions = []
    for edge_dict in edge_dict_list:
        pixels = region_grow_constraint2(edge_dict, img)
        regions.append(pixels)

    # 4) Build labeled visualization mask
    mask = np.zeros(img.shape, dtype=np.int32)
    label = 5
    for region_coords in regions:
        for y, x in region_coords:
            mask[y, x] = label
        label += 1

    plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap="gray")
    plt.imshow(mask, cmap="nipy_spectral", alpha=0.5)
    plt.show()

    # 5) Save binary and intensity-masked outputs
    binary_mask = (mask > 0).astype(np.uint8) * 255
    tiff.imwrite("/segmented_mask.tif", binary_mask)

    seg_result = img.copy().astype(np.float32)
    seg_result[mask == 0] = 0.0
    tiff.imwrite("/segmented_region.tif", seg_result)
