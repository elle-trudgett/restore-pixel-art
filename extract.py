#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pillow",
#     "scipy",
# ]
# ///
"""
Extract clean pixel art from AI-generated upscaled pixel art images.

The input is expected to be pixel art where each "logical pixel" is rendered
as a larger block with potentially soft/blurred edges. Cell sizes may vary
throughout the image. This script detects the actual grid lines adaptively,
extracts the dominant color from each cell, and outputs a clean 1:1 pixel art image.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d


def color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Simple RGB Euclidean distance, normalized to 0-~441 range."""
    dr = int(c1[0]) - int(c2[0])
    dg = int(c1[1]) - int(c2[1])
    db = int(c1[2]) - int(c2[2])
    return np.sqrt(dr * dr + dg * dg + db * db)


def quantize_to_n_colors(
    colors: list[tuple[int, int, int]],
    n: int,
) -> list[tuple[int, int, int]]:
    """
    Quantize to n colors using greedy selection.
    First clusters similar colors, then selects based on cluster size.
    """
    if len(colors) <= n:
        return list(set(colors))

    unique_colors = list(set(colors))

    # First, cluster similar colors together (within distance 20)
    clusters = []  # List of (representative_color, count, member_colors)
    cluster_threshold = 20.0

    for color in colors:
        # Find if this color belongs to an existing cluster
        found_cluster = False
        for i, (rep, count, members) in enumerate(clusters):
            if color_distance(color, rep) < cluster_threshold:
                # Add to this cluster, update representative to most common
                members.append(color)
                clusters[i] = (rep, count + 1, members)
                found_cluster = True
                break

        if not found_cluster:
            # Start new cluster
            clusters.append((color, 1, [color]))

    # Sort clusters by size (largest first)
    clusters.sort(key=lambda x: x[1], reverse=True)

    # For each cluster, pick the most common actual color as representative
    from collections import Counter
    cluster_reps = []
    for rep, count, members in clusters:
        most_common = Counter(members).most_common(1)[0][0]
        cluster_reps.append((most_common, count))

    # Greedy selection from cluster representatives
    palette = [cluster_reps[0][0]]

    for color, count in cluster_reps[1:]:
        if len(palette) >= n:
            break
        # Check if distinct from existing palette
        min_dist = min(color_distance(color, p) for p in palette)
        if min_dist > 15.0:  # Ensure palette colors are distinct
            palette.append(color)

    # Fill remaining slots by covering gaps (colors with highest error)
    if len(palette) < n:
        unique = list(set(colors))
        for _ in range(n - len(palette)):
            # Find the color with highest error to current palette
            worst_color = None
            worst_error = 0
            for c in unique:
                err = min(color_distance(c, p) for p in palette)
                if err > worst_error:
                    worst_error = err
                    worst_color = c
            if worst_color and worst_error > 10:  # Only add if there's a meaningful gap
                palette.append(worst_color)
                unique.remove(worst_color)
            else:
                break

    return palette


def compute_max_error(
    original_colors: list[tuple[int, int, int]],
    palette: list[tuple[int, int, int]],
) -> float:
    """Compute the maximum color distance from any original color to its nearest palette color."""
    max_err = 0
    for c in original_colors:
        min_dist = min(color_distance(c, p) for p in palette)
        max_err = max(max_err, min_dist)
    return max_err


def extract_palette(
    colors: list[tuple[int, int, int, int]],
    max_error: float = 15.0,
    verbose: bool = False,
) -> list[tuple[int, int, int, int]]:
    """
    Find minimum palette size that keeps max color error below threshold.
    Uses binary search over palette sizes.
    """
    # Get unique RGB colors
    unique_rgb = list(set((r, g, b) for r, g, b, a in colors))

    if verbose:
        print(f"  {len(unique_rgb)} unique colors from {len(colors)} samples")

    if len(unique_rgb) <= 2:
        return [(r, g, b, 255) for r, g, b in unique_rgb]

    # Binary search for minimum palette size
    lo, hi = 2, min(128, len(unique_rgb))

    # First check if we even need to reduce
    if len(unique_rgb) <= hi:
        err = compute_max_error(unique_rgb, unique_rgb)
        if err <= max_error:
            return [(r, g, b, 255) for r, g, b in unique_rgb]

    best_palette = unique_rgb
    best_size = len(unique_rgb)

    while lo <= hi:
        mid = (lo + hi) // 2
        palette = quantize_to_n_colors(unique_rgb, mid)
        err = compute_max_error(unique_rgb, palette)

        if verbose:
            print(f"  Trying {mid} colors: max_error={err:.2f}")

        if err <= max_error:
            # Can reduce further
            best_palette = palette
            best_size = mid
            hi = mid - 1
        else:
            # Need more colors
            lo = mid + 1

    if verbose:
        print(f"  Final palette: {len(best_palette)} colors")

    return [(r, g, b, 255) for r, g, b in best_palette]


def snap_to_palette(color: tuple[int, int, int, int], palette: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Find the nearest palette color using perceptual distance."""
    best_dist = float('inf')
    best_color = color

    r, g, b, a = color
    for pr, pg, pb, pa in palette:
        dist = color_distance((r, g, b), (pr, pg, pb))
        if dist < best_dist:
            best_dist = dist
            best_color = (pr, pg, pb, a)  # Keep original alpha

    return best_color


def snap_colors_spatially(
    pixels: list[tuple[int, int, tuple[int, int, int, int]]],  # (col, row, color)
    palette: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    spatial_weight: float = 0.3,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """
    Snap colors to palette considering spatial neighbors.
    Nearby pixels prefer using the same palette colors.
    """
    # Build grid of original colors
    grid = {}
    for col, row, color in pixels:
        grid[(col, row)] = color

    # First pass: snap each pixel to nearest palette color
    snapped = {}
    for col, row, color in pixels:
        snapped[(col, row)] = snap_to_palette(color, palette)

    # Second pass: refine by considering neighbors
    # Iterate a few times to propagate spatial consistency
    for _ in range(3):
        new_snapped = {}
        for col, row, original_color in pixels:
            r, g, b, a = original_color

            # Score each palette color
            best_score = float('inf')
            best_color = snapped[(col, row)]

            for pc in palette:
                # Color distance to original
                color_dist = color_distance((r, g, b), (pc[0], pc[1], pc[2]))

                # Spatial term: bonus if neighbors use this color
                neighbor_bonus = 0
                neighbor_count = 0
                for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    nc, nr = col + dc, row + dr
                    if (nc, nr) in snapped:
                        neighbor_count += 1
                        if snapped[(nc, nr)][:3] == pc[:3]:
                            neighbor_bonus += 1

                # Combined score (lower is better)
                if neighbor_count > 0:
                    spatial_term = -spatial_weight * (neighbor_bonus / neighbor_count) * 20
                else:
                    spatial_term = 0

                score = color_dist + spatial_term

                if score < best_score:
                    best_score = score
                    best_color = (pc[0], pc[1], pc[2], a)

            new_snapped[(col, row)] = best_color

        snapped = new_snapped

    return snapped


def compute_local_variance(img_array: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Compute local color variance at each pixel."""
    from scipy.ndimage import uniform_filter

    if len(img_array.shape) == 3:
        # Compute variance across color channels
        img_float = img_array[:, :, :3].astype(float)
        mean = uniform_filter(img_float, size=(window_size, window_size, 1))
        sq_mean = uniform_filter(img_float ** 2, size=(window_size, window_size, 1))
        variance = np.sum(sq_mean - mean ** 2, axis=2)
    else:
        img_float = img_array.astype(float)
        mean = uniform_filter(img_float, size=window_size)
        sq_mean = uniform_filter(img_float ** 2, size=window_size)
        variance = sq_mean - mean ** 2

    return variance


def find_pixel_centers_1d(
    variance_signal: np.ndarray,
    min_cell_size: int,
    max_cell_size: int,
) -> list[int]:
    """
    Find pixel centers along one axis by looking for valleys in variance.
    Low variance = solid pixel center, high variance = boundary.
    """
    # Smooth the signal slightly
    smoothed = gaussian_filter1d(variance_signal, sigma=0.5)

    # Find local minima (pixel centers)
    centers = []
    pos = 0

    while pos < len(smoothed):
        # Search for the next local minimum within the valid range
        search_end = min(pos + max_cell_size + 1, len(smoothed))

        if pos == 0:
            # For the first pixel, find the first minimum
            search_start = 0
        else:
            search_start = pos + min_cell_size - 1

        if search_start >= len(smoothed):
            break

        search_region = smoothed[search_start:search_end]
        if len(search_region) == 0:
            break

        # Find the minimum variance point (most likely pixel center)
        local_min_idx = np.argmin(search_region)
        center = search_start + local_min_idx

        centers.append(center)

        # Move to search for next pixel
        # Next pixel should be at least min_cell_size away
        pos = center + 1

    return centers


def estimate_cell_size(img_array: np.ndarray, min_cell_size: int, max_cell_size: int) -> tuple[int, int]:
    """Estimate the typical cell size by analyzing variance patterns."""
    variance = compute_local_variance(img_array, window_size=3)

    # Project variance onto each axis
    v_variance = np.mean(variance, axis=0)  # variance along x
    h_variance = np.mean(variance, axis=1)  # variance along y

    # Find peaks in variance (boundaries between pixels)
    def find_peak_spacing(signal, min_size, max_size):
        smoothed = gaussian_filter1d(signal, sigma=1)
        threshold = np.percentile(smoothed, 70)

        # Find peaks above threshold
        peaks = []
        in_peak = False
        peak_start = 0

        for i, v in enumerate(smoothed):
            if v > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif v <= threshold and in_peak:
                in_peak = False
                peaks.append((peak_start + i) // 2)

        if len(peaks) < 2:
            return (min_size + max_size) // 2

        gaps = np.diff(peaks)
        valid_gaps = gaps[(gaps >= min_size) & (gaps <= max_size)]

        if len(valid_gaps) == 0:
            return (min_size + max_size) // 2

        return int(np.median(valid_gaps))

    cell_w = find_peak_spacing(v_variance, min_cell_size, max_cell_size)
    cell_h = find_peak_spacing(h_variance, min_cell_size, max_cell_size)

    return cell_w, cell_h


def find_grid_lines(
    img_array: np.ndarray,
    axis: int,
    target_cell_size: int,
    tolerance: float = 0.3,
    edge_threshold_percentile: float = 80,
) -> list[int]:
    """
    Find grid lines along an axis, using the target cell size to filter outliers.

    Args:
        img_array: Input image as numpy array
        axis: 0 for horizontal lines (row boundaries), 1 for vertical lines (column boundaries)
        target_cell_size: Expected cell size (dominant size detected)
        tolerance: How much variance from target to allow (0.3 = 30%)
        edge_threshold_percentile: Percentile for edge strength threshold

    Returns:
        List of positions where grid lines are detected
    """
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2)
    else:
        gray = img_array.astype(float)

    gradient = np.abs(np.diff(gray, axis=axis))
    edge_signal = np.sum(gradient, axis=1 - axis)
    # Use smaller sigma for small cells to preserve fine edges
    sigma = 0.5 if target_cell_size <= 5 else 1.0
    edge_signal = gaussian_filter1d(edge_signal, sigma=sigma)

    threshold = np.percentile(edge_signal, edge_threshold_percentile)
    candidates = np.where(edge_signal > threshold)[0]

    size = gray.shape[axis]

    if len(candidates) == 0:
        return list(range(0, size + 1, target_cell_size))

    min_cell = int(target_cell_size * (1 - tolerance))
    max_cell = int(target_cell_size * (1 + tolerance))

    # Cluster nearby candidates (scale gap with cell size)
    cluster_gap = max(1, min_cell // 3)
    clusters: list[list[int]] = []
    current_cluster = [candidates[0]]

    for i in range(1, len(candidates)):
        if candidates[i] - candidates[i - 1] < cluster_gap:
            current_cluster.append(candidates[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [candidates[i]]
    clusters.append(current_cluster)

    # Get cluster centers weighted by edge strength
    cluster_centers = []
    for cluster in clusters:
        weights = edge_signal[cluster]
        center = int(np.average(cluster, weights=weights))
        cluster_centers.append((center, np.max(weights)))

    # Build grid starting from 0, selecting edges that match expected spacing
    grid_lines = [0]

    while grid_lines[-1] < size - min_cell // 2:
        current = grid_lines[-1]
        expected_next = current + target_cell_size

        # Find candidates in the acceptable range
        valid_candidates = [
            (pos, strength) for pos, strength in cluster_centers
            if min_cell <= pos - current <= max_cell
        ]

        if valid_candidates:
            # Prefer the one closest to expected position, weighted by edge strength
            best = min(
                valid_candidates,
                key=lambda x: abs(x[0] - expected_next) / (1 + x[1] / np.max(edge_signal))
            )
            grid_lines.append(best[0])
        else:
            # No good candidate - extrapolate using target size
            next_pos = current + target_cell_size
            if next_pos < size:
                grid_lines.append(next_pos)
            else:
                break

    # Ensure we include the end
    if size - grid_lines[-1] > min_cell // 2:
        grid_lines.append(size)

    return grid_lines


def compute_edge_signal(img_array: np.ndarray, axis: int) -> np.ndarray:
    """Compute edge strength along an axis (sum of gradients)."""
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:, :, :3], axis=2)
    else:
        gray = img_array.astype(float)
    gradient = np.abs(np.diff(gray, axis=axis))
    return np.sum(gradient, axis=1 - axis)


def score_grid_alignment(edge_signal: np.ndarray, cell_size: float, offset: float = 0) -> float:
    """
    Score how well grid lines align with edges.
    Boundaries should have high edge strength, centers should have low.
    """
    length = len(edge_signal)
    n_cells = int((length - offset) / cell_size)
    if n_cells < 2:
        return 0

    boundary_score = 0
    center_score = 0

    for i in range(1, n_cells):
        # Boundary position
        boundary = int(offset + i * cell_size)
        if 0 <= boundary < length:
            boundary_score += edge_signal[boundary]

        # Center position
        center = int(offset + (i - 0.5) * cell_size)
        if 0 <= center < length:
            center_score += edge_signal[center]

    # Want high boundaries, low centers
    return boundary_score - center_score * 0.5


def find_period_autocorr(signal: np.ndarray, min_period: int = 3, max_period: int = 50) -> int:
    """Find the dominant period in a signal using autocorrelation."""
    # Normalize signal
    signal = signal - np.mean(signal)
    if np.std(signal) > 0:
        signal = signal / np.std(signal)

    # Compute autocorrelation
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]  # Take positive lags only

    # Find peaks in autocorrelation (excluding lag 0)
    best_period = min_period
    best_score = 0

    for period in range(min_period, min(max_period + 1, n // 2)):
        score = autocorr[period]
        if score > best_score:
            best_score = score
            best_period = period

    return best_period


def find_best_grid(img_array: np.ndarray, min_period: int = 3, max_period: int = 20) -> tuple[int, int, float, float]:
    """
    Find the best grid by optimizing cell size AND offset.
    Returns (n_cols, n_rows, offset_x, offset_y).
    """
    height, width = img_array.shape[:2]

    h_edges = compute_edge_signal(img_array, axis=0)  # horizontal edges -> rows
    v_edges = compute_edge_signal(img_array, axis=1)  # vertical edges -> cols

    # Get initial estimate from autocorrelation
    cell_w = find_period_autocorr(v_edges, min_period, max_period)
    cell_h = find_period_autocorr(h_edges, min_period, max_period)

    # Pixels should be roughly square
    if cell_w > 0 and cell_h > 0:
        ratio = max(cell_w, cell_h) / min(cell_w, cell_h)
        if ratio > 1.5:
            cell_size = min(cell_w, cell_h)
            cell_w = cell_h = cell_size

    # Optimize X and Y independently (much faster)
    best_score_x = -float('inf')
    best_cw, best_ox = cell_w, 0.0

    for cw in np.linspace(cell_w * 0.85, cell_w * 1.15, 31):
        for ox in np.linspace(0, cw - 0.1, 21):
            score = score_grid_alignment(v_edges, cw, ox)
            if score > best_score_x:
                best_score_x = score
                best_cw, best_ox = cw, ox

    best_score_y = -float('inf')
    best_ch, best_oy = cell_h, 0.0

    for ch in np.linspace(cell_h * 0.85, cell_h * 1.15, 31):
        for oy in np.linspace(0, ch - 0.1, 21):
            score = score_grid_alignment(h_edges, ch, oy)
            if score > best_score_y:
                best_score_y = score
                best_ch, best_oy = ch, oy

    cell_w, cell_h, offset_x, offset_y = best_cw, best_ch, best_ox, best_oy

    n_cols = int((width - offset_x) / cell_w) + (1 if offset_x > 0 else 0)
    n_rows = int((height - offset_y) / cell_h) + (1 if offset_y > 0 else 0)

    return max(1, n_cols), max(1, n_rows), offset_x, offset_y, cell_w, cell_h


def extract_pixels_by_centers(
    img_array: np.ndarray,
    min_cell_size: int,
    max_cell_size: int,
    verbose: bool = True,
) -> tuple[Image.Image, list[tuple[int, int]], int, int]:
    """
    Extract pixel art by finding pixel centers directly.
    Returns (output_image, centers, width, height).
    """
    centers = find_pixel_centers_2d(img_array, min_cell_size, max_cell_size)

    if not centers:
        raise ValueError("Could not find any pixel centers")

    # Determine grid dimensions from center positions
    xs = sorted(set(c[0] for c in centers))
    ys = sorted(set(c[1] for c in centers))

    # Cluster x and y coordinates to find grid lines
    def cluster_coords(coords, min_gap):
        if not coords:
            return []
        clusters = [[coords[0]]]
        for c in coords[1:]:
            if c - clusters[-1][-1] < min_gap:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [int(np.mean(cl)) for cl in clusters]

    x_grid = cluster_coords(xs, min_cell_size // 2)
    y_grid = cluster_coords(ys, min_cell_size // 2)

    if verbose:
        print(f"Found {len(x_grid)} columns, {len(y_grid)} rows")

    out_width = len(x_grid)
    out_height = len(y_grid)

    output = Image.new("RGBA", (out_width, out_height))

    # Create lookup for grid positions
    def find_grid_idx(val, grid, tolerance):
        for i, g in enumerate(grid):
            if abs(val - g) <= tolerance:
                return i
        return -1

    # Map centers to grid positions and sample colors
    tolerance = max_cell_size // 2
    grid_colors = {}

    for cx, cy in centers:
        gx = find_grid_idx(cx, x_grid, tolerance)
        gy = find_grid_idx(cy, y_grid, tolerance)
        if gx >= 0 and gy >= 0 and (gx, gy) not in grid_colors:
            # Sample color at this center
            pixel = img_array[cy, cx]
            if len(pixel) >= 4:
                color = (int(pixel[0]), int(pixel[1]), int(pixel[2]), int(pixel[3]))
            elif len(pixel) >= 3:
                color = (int(pixel[0]), int(pixel[1]), int(pixel[2]), 255)
            else:
                color = (int(pixel[0]), int(pixel[0]), int(pixel[0]), 255)
            grid_colors[(gx, gy)] = color

    # Fill output image
    for gy in range(out_height):
        for gx in range(out_width):
            if (gx, gy) in grid_colors:
                output.putpixel((gx, gy), grid_colors[(gx, gy)])
            else:
                # Fallback: interpolate from nearest center
                output.putpixel((gx, gy), (0, 0, 0, 255))

    return output, list(zip(x_grid, y_grid)), out_width, out_height


def save_debug_centers(
    img: Image.Image,
    centers: list[tuple[int, int]],
    output_path: str | Path,
) -> None:
    """Save a copy of the image with detected pixel centers marked."""
    from PIL import ImageDraw
    debug_img = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(debug_img)

    # Draw a small cross at each center
    for x, y in centers:
        draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], fill=(255, 0, 0, 255))

    debug_img.save(output_path)


def extract_pixel_art(
    input_path: str | Path,
    output_path: str | Path | None = None,
    verbose: bool = True,
    debug: bool = False,
) -> Image.Image:
    """
    Extract clean pixel art from an upscaled pixel art image.

    Args:
        input_path: Path to the input image
        output_path: Path to save the output (optional)
        verbose: Print detection info
        debug: Save a debug image showing detected pixel centers

    Returns:
        The extracted pixel art image
    """
    # Load image
    img = Image.open(input_path)
    img_array = np.array(img)

    if verbose:
        print(f"Input image: {img.size[0]}x{img.size[1]}")

    # Find the best grid dimensions and offset by minimizing alignment error
    n_cols, n_rows, offset_x, offset_y, cell_w, cell_h = find_best_grid(img_array)

    if verbose:
        print(f"Best grid: {n_cols}x{n_rows} (cell size ~{cell_w:.1f}x{cell_h:.1f}px, offset {offset_x:.1f},{offset_y:.1f})")

    out_width = n_cols
    out_height = n_rows

    # Use strict grid centers with offset
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            cx = int(offset_x + (col + 0.5) * cell_w)
            cy = int(offset_y + (row + 0.5) * cell_h)

            # Clamp to image bounds
            cx = max(0, min(img.size[0] - 1, cx))
            cy = max(0, min(img.size[1] - 1, cy))

            centers.append((col, row, cx, cy))

    # Save debug overlay if requested
    if debug and output_path:
        debug_path = Path(output_path).parent / f"{Path(output_path).stem}_centers.png"
        save_debug_centers(img, [(c[2], c[3]) for c in centers], debug_path)
        if verbose:
            print(f"Debug centers saved to: {debug_path}")

    if out_width <= 0 or out_height <= 0:
        raise ValueError("Could not detect valid pixel grid.")

    if verbose:
        print(f"Output size: {out_width}x{out_height}")

    # Create output image
    output = Image.new("RGBA", (out_width, out_height))

    # First pass: sample colors from small region around center, pick most common
    sampled_colors = []
    sample_radius = max(1, int(min(cell_w, cell_h) * 0.2))  # 20% of cell size

    for col, row, cx, cy in centers:
        # Sample a small region
        x1 = max(0, cx - sample_radius)
        x2 = min(img.size[0], cx + sample_radius + 1)
        y1 = max(0, cy - sample_radius)
        y2 = min(img.size[1], cy + sample_radius + 1)

        region = img_array[y1:y2, x1:x2]

        if region.size > 0:
            # Flatten to list of colors and find most common
            pixels = region.reshape(-1, region.shape[-1]) if len(region.shape) == 3 else region.reshape(-1, 1)

            # Convert to tuples for counting
            from collections import Counter
            if pixels.shape[1] >= 3:
                color_tuples = [tuple(p[:3]) for p in pixels]
            else:
                color_tuples = [(int(p[0]), int(p[0]), int(p[0])) for p in pixels]

            most_common = Counter(color_tuples).most_common(1)[0][0]
            color = (most_common[0], most_common[1], most_common[2], 255)
        else:
            pixel = img_array[cy, cx]
            if len(pixel) >= 3:
                color = (int(pixel[0]), int(pixel[1]), int(pixel[2]), 255)
            else:
                color = (int(pixel[0]), int(pixel[0]), int(pixel[0]), 255)

        sampled_colors.append((col, row, color))

    # Extract palette and snap colors
    all_colors = [c[2] for c in sampled_colors]
    palette = extract_palette(all_colors, verbose=verbose)

    if verbose:
        print(f"Palette: {len(palette)} colors")
        # Show palette as hex colors
        hex_colors = ['#' + ''.join(f'{c:02x}' for c in p[:3]) for p in palette]
        print(f"  Colors: {', '.join(hex_colors)}")

    # First pass: snap all colors to palette
    snapped_grid = {}
    original_grid = {}
    for col, row, color in sampled_colors:
        snapped = snap_to_palette(color, palette)
        snapped_grid[(col, row)] = snapped
        original_grid[(col, row)] = color

    # Write to output (no spatial smoothing for now)
    for (col, row), snapped in snapped_grid.items():
        output.putpixel((col, row), snapped)

    # Save if output path provided
    if output_path:
        output.save(output_path)
        if verbose:
            print(f"Saved to: {output_path}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clean pixel art from AI-generated upscaled images"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path (default: input_pixels.png)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--debug", action="store_true", help="Save debug image with detected centers")

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_pixels.png"

    extract_pixel_art(
        args.input,
        args.output,
        verbose=not args.quiet,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
