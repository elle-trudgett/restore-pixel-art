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


def compute_palette_quality(
    colors: list[tuple[int, int, int]],
    palette: list[tuple[int, int, int]],
    soft_threshold: float,
) -> tuple[float, float]:
    """
    Compute palette quality metrics.
    Returns (deviation_rate, max_error) where:
    - deviation_rate: fraction of colors beyond soft_threshold
    - max_error: maximum distance of any color to its nearest palette color
    """
    if not colors or not palette:
        return 1.0, float('inf')
    deviating = 0
    max_err = 0.0
    for c in colors:
        min_dist = min(color_distance(c, p) for p in palette)
        if min_dist > soft_threshold:
            deviating += 1
        max_err = max(max_err, min_dist)
    return deviating / len(colors), max_err


def precluster_colors_spatial(
    sampled_colors: list[tuple[int, int, tuple[int, int, int, int]]],
    neighbor_threshold: float = 8.0,
    chain_threshold: float = 12.0,
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """
    Walk the image spatially and build chains of similar adjacent pixels.
    Each pixel must be close to its neighbor AND not too far from chain average.

    Args:
        sampled_colors: List of (col, row, color) tuples
        neighbor_threshold: Max distance between adjacent pixels to chain
        chain_threshold: Max distance from chain average to join

    Returns:
        List of (col, row, unified_color) where unified_color is chain representative
    """
    # Build lookup by position
    color_grid = {(col, row): color for col, row, color in sampled_colors}
    max_col = max(col for col, row, _ in sampled_colors)
    max_row = max(row for col, row, _ in sampled_colors)

    # Track which pixels are assigned to chains
    chain_id = {}  # (col, row) -> chain index
    chains = []  # List of [(col, row), ...] for each chain
    chain_colors = []  # Running average color for each chain

    def get_rgb(color):
        return (color[0], color[1], color[2])

    def rgb_avg(colors_list):
        if not colors_list:
            return (0, 0, 0)
        r = sum(c[0] for c in colors_list) // len(colors_list)
        g = sum(c[1] for c in colors_list) // len(colors_list)
        b = sum(c[2] for c in colors_list) // len(colors_list)
        return (r, g, b)

    # Process pixels in scan order
    for row in range(max_row + 1):
        for col in range(max_col + 1):
            if (col, row) not in color_grid:
                continue

            color = color_grid[(col, row)]
            rgb = get_rgb(color)

            # Check neighbors (left and up, since we scan left-to-right, top-to-bottom)
            best_chain = None
            best_dist = float('inf')

            for dc, dr in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                nc, nr = col + dc, row + dr
                if (nc, nr) in chain_id:
                    neighbor_color = color_grid[(nc, nr)]
                    neighbor_rgb = get_rgb(neighbor_color)
                    cid = chain_id[(nc, nr)]
                    chain_avg = chain_colors[cid]

                    # Check both: close to neighbor AND close to chain average
                    neighbor_dist = color_distance(rgb, neighbor_rgb)
                    chain_dist = color_distance(rgb, chain_avg)

                    if neighbor_dist < neighbor_threshold and chain_dist < chain_threshold:
                        if neighbor_dist < best_dist:
                            best_dist = neighbor_dist
                            best_chain = cid

            if best_chain is not None:
                # Join existing chain
                chain_id[(col, row)] = best_chain
                chains[best_chain].append((col, row))
                # Update running average
                chain_rgbs = [get_rgb(color_grid[pos]) for pos in chains[best_chain]]
                chain_colors[best_chain] = rgb_avg(chain_rgbs)
            else:
                # Start new chain
                new_id = len(chains)
                chain_id[(col, row)] = new_id
                chains.append([(col, row)])
                chain_colors.append(rgb)

    # Merge chains with similar colors
    merge_threshold = chain_threshold * 0.8  # Slightly tighter than chain threshold
    chain_map = list(range(len(chains)))  # chain_map[i] = canonical chain for i

    def find_canonical(i):
        while chain_map[i] != i:
            i = chain_map[i]
        return i

    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            ci, cj = find_canonical(i), find_canonical(j)
            if ci != cj:
                dist = color_distance(chain_colors[ci], chain_colors[cj])
                if dist < merge_threshold:
                    # Merge j into i
                    chain_map[cj] = ci

    # Compute merged colors (average of merged chains)
    merged_colors = {}
    for i in range(len(chains)):
        ci = find_canonical(i)
        if ci not in merged_colors:
            # Find all chains that merge to this one
            member_chains = [j for j in range(len(chains)) if find_canonical(j) == ci]
            # Average their colors weighted by chain size
            total_pixels = sum(len(chains[j]) for j in member_chains)
            r = sum(chain_colors[j][0] * len(chains[j]) for j in member_chains) // total_pixels
            g = sum(chain_colors[j][1] * len(chains[j]) for j in member_chains) // total_pixels
            b = sum(chain_colors[j][2] * len(chains[j]) for j in member_chains) // total_pixels
            merged_colors[ci] = (r, g, b)

    # Build result with unified colors
    result = []
    for col, row, color in sampled_colors:
        if (col, row) in chain_id:
            cid = chain_id[(col, row)]
            canonical = find_canonical(cid)
            avg = merged_colors[canonical]
            result.append((col, row, (avg[0], avg[1], avg[2], color[3])))
        else:
            result.append((col, row, color))

    return result


def extract_palette_quality(
    colors: list[tuple[int, int, int, int]],
    max_deviation: float = 12.0,  # No pixel should differ more than this
    soft_threshold: float = 5.0,  # Threshold for "noticeable" deviation
    max_deviation_rate: float = 0.05,  # Max 5% of pixels can exceed soft_threshold
    precluster_threshold: float = 15.0,  # Merge colors within this distance first
    verbose: bool = False,
) -> list[tuple[int, int, int, int]]:
    """
    Find smallest palette where:
    1. No pixel differs by more than max_deviation (hard limit)
    2. At most max_deviation_rate of pixels differ by more than soft_threshold

    Tries all palette sizes linearly to handle non-monotonic quality
    (where N+1 colors might be worse than N due to different clustering).
    """
    # Pre-cluster to reduce AI noise
    if precluster_threshold > 0:
        colors = precluster_colors(colors, precluster_threshold)

    unique_rgb = list(set((r, g, b) for r, g, b, a in colors))
    all_rgb = [(r, g, b) for r, g, b, a in colors]

    if verbose:
        print(f"  {len(unique_rgb)} unique colors from {len(colors)} samples")

    if len(unique_rgb) <= 2:
        return [(r, g, b, 255) for r, g, b in unique_rgb]

    # Compute all palette sizes first
    min_size, max_size = 10, min(80, len(unique_rgb))
    all_results = []  # (size, palette, dev_rate, max_err)

    for n in range(min_size, max_size + 1):
        palette = quantize_to_n_colors(unique_rgb, n)
        dev_rate, max_err = compute_palette_quality(all_rgb, palette, soft_threshold=soft_threshold)
        all_results.append((n, palette, dev_rate, max_err))

        if verbose:
            passes_max = max_err <= max_deviation
            passes_rate = dev_rate <= max_deviation_rate
            marker = "✓" if (passes_max and passes_rate) else ""
            print(f"  {n} colors: max_err={max_err:.1f}, dev_rate={dev_rate:.1%} {marker}")

    # Pick smallest palette that meets both thresholds
    valid = [(n, p, dr, e) for n, p, dr, e in all_results
             if e <= max_deviation and dr <= max_deviation_rate]

    if valid:
        best_size, best_palette, best_dr, best_err = min(valid, key=lambda x: x[0])
        if verbose:
            print(f"  Selected: {best_size} colors (max_err={best_err:.1f}, dev_rate={best_dr:.1%})")
        return [(r, g, b, 255) for r, g, b in best_palette]

    # Nothing met both thresholds - try just max_err
    valid_max = [(n, p, dr, e) for n, p, dr, e in all_results if e <= max_deviation]
    if valid_max:
        best_size, best_palette, best_dr, best_err = min(valid_max, key=lambda x: x[0])
        if verbose:
            print(f"  Selected: {best_size} colors (max_err={best_err:.1f}, dev_rate={best_dr:.1%} - rate exceeded)")
        return [(r, g, b, 255) for r, g, b in best_palette]

    # Nothing met threshold - use largest
    if verbose:
        print(f"  Warning: no palette met threshold {max_deviation}, using max size")
    return [(r, g, b, 255) for r, g, b in all_results[-1][1]]


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


def find_period_autocorr(signal: np.ndarray, min_period: int = 3, max_period: int = 100) -> int:
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


def find_edge_position(
    img_array: np.ndarray,
    start_x: int,
    start_y: int,
    dx: int,
    dy: int,
    max_dist: int,
    min_edge_strength: float = 15.0,
) -> tuple[int | None, float]:
    """
    Scan from start position in direction (dx, dy) to find the strongest edge.
    Returns (distance to max gradient, max gradient strength).
    """
    height, width = img_array.shape[:2]

    best_dist = None
    best_strength = 0.0

    if len(img_array.shape) == 3:
        prev_color = img_array[start_y, start_x, :3].astype(float)
    else:
        prev_color = np.array([img_array[start_y, start_x]]).astype(float)

    for dist in range(1, max_dist + 1):
        x = start_x + dx * dist
        y = start_y + dy * dist

        if not (0 <= x < width and 0 <= y < height):
            break

        if len(img_array.shape) == 3:
            curr_color = img_array[y, x, :3].astype(float)
        else:
            curr_color = np.array([img_array[y, x]]).astype(float)

        color_diff = np.sqrt(np.sum((curr_color - prev_color) ** 2))

        if color_diff > best_strength:
            best_strength = color_diff
            best_dist = dist

        prev_color = curr_color

    # Only return if edge is strong enough
    if best_strength >= min_edge_strength:
        return best_dist, best_strength
    return None, 0.0


def refine_center_by_edges(
    img_array: np.ndarray,
    cx: int,
    cy: int,
    cell_w: float,
    cell_h: float,
) -> tuple[int, int]:
    """
    Refine center by finding actual edge positions in all 4 directions
    and computing a weighted midpoint based on edge strength.
    """
    max_dist = int(max(cell_w, cell_h) * 0.7)

    # Find edges in all 4 directions (now returns strength too)
    left_dist, left_str = find_edge_position(img_array, cx, cy, -1, 0, max_dist)
    right_dist, right_str = find_edge_position(img_array, cx, cy, 1, 0, max_dist)
    up_dist, up_str = find_edge_position(img_array, cx, cy, 0, -1, max_dist)
    down_dist, down_str = find_edge_position(img_array, cx, cy, 0, 1, max_dist)

    new_cx, new_cy = float(cx), float(cy)

    # Horizontal: use both edges if found, or infer from one strong edge + expected cell width
    if left_dist is not None and right_dist is not None:
        # Both edges found - use simple midpoint (edge strength determines trust, not position)
        left_edge = cx - left_dist
        right_edge = cx + right_dist
        new_cx = (left_edge + right_edge) / 2
    elif left_dist is not None and left_str > 30:
        # Only left edge found (strong) - estimate right from cell width
        left_edge = cx - left_dist
        new_cx = left_edge + cell_w / 2
    elif right_dist is not None and right_str > 30:
        # Only right edge found (strong) - estimate left from cell width
        right_edge = cx + right_dist
        new_cx = right_edge - cell_w / 2

    # Vertical: same logic
    if up_dist is not None and down_dist is not None:
        # Both edges found - use simple midpoint
        up_edge = cy - up_dist
        down_edge = cy + down_dist
        new_cy = (up_edge + down_edge) / 2
    elif up_dist is not None and up_str > 30:
        up_edge = cy - up_dist
        new_cy = up_edge + cell_h / 2
    elif down_dist is not None and down_str > 30:
        down_edge = cy + down_dist
        new_cy = down_edge - cell_h / 2

    return int(new_cx), int(new_cy)


def compute_center_score(
    img_array: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> float:
    """
    Score a candidate center point. Lower is better.

    Considers:
    - Color variance in neighborhood (lower = better, more uniform)
    - Edge strength at point (lower = better, away from boundaries)
    """
    height, width = img_array.shape[:2]

    # Clamp to bounds
    x1 = max(0, cx - radius)
    x2 = min(width, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(height, cy + radius + 1)

    region = img_array[y1:y2, x1:x2]
    if region.size == 0:
        return float('inf')

    # Color variance penalty (want low variance = uniform color)
    if len(region.shape) == 3:
        pixels = region.reshape(-1, 3).astype(float)
    else:
        pixels = region.reshape(-1, 1).astype(float)

    variance = np.mean(np.var(pixels, axis=0))

    # Edge strength penalty (want low edge = center of pixel, not boundary)
    # Compute gradient magnitude at center
    if 0 < cx < width - 1 and 0 < cy < height - 1:
        if len(img_array.shape) == 3:
            center_pixel = img_array[cy, cx, :3].astype(float)
            neighbors = [
                img_array[cy-1, cx, :3].astype(float),
                img_array[cy+1, cx, :3].astype(float),
                img_array[cy, cx-1, :3].astype(float),
                img_array[cy, cx+1, :3].astype(float),
            ]
        else:
            center_pixel = np.array([img_array[cy, cx]]).astype(float)
            neighbors = [
                np.array([img_array[cy-1, cx]]).astype(float),
                np.array([img_array[cy+1, cx]]).astype(float),
                np.array([img_array[cy, cx-1]]).astype(float),
                np.array([img_array[cy, cx+1]]).astype(float),
            ]

        edge_strength = np.mean([np.sqrt(np.sum((center_pixel - n) ** 2)) for n in neighbors])
    else:
        edge_strength = 0

    # Combined score (lower is better)
    return variance + edge_strength * 2


def find_edge_bounds(
    img_array: np.ndarray,
    start_x: int,
    start_y: int,
    dx: int,
    dy: int,
    center_color: np.ndarray,
    max_dist: int,
    color_threshold: float = 25.0,
) -> tuple[int | None, int | None]:
    """
    Scan from start position to find edge transition zone.
    Returns (last_stable_of_my_color, first_stable_of_next_color) as distances.
    Returns (None, None) if no edge found (same color continues).
    """
    height, width = img_array.shape[:2]

    last_stable_mine = 0  # Last position still matching center color
    first_stable_next = None  # First position that's stable at a different color

    in_transition = False
    transition_start = None

    for dist in range(1, max_dist + 1):
        x = start_x + dx * dist
        y = start_y + dy * dist

        if not (0 <= x < width and 0 <= y < height):
            break

        if len(img_array.shape) == 3:
            pixel_color = img_array[y, x, :3].astype(float)
        else:
            pixel_color = np.array([img_array[y, x]]).astype(float)

        color_diff = np.sqrt(np.sum((pixel_color - center_color) ** 2))

        if color_diff < color_threshold:
            # Still my color
            last_stable_mine = dist
            in_transition = False
            transition_start = None
        else:
            # Different color - are we in transition or stable?
            if not in_transition:
                in_transition = True
                transition_start = dist

            # If color has changed drastically from center, treat as "definitely next color"
            # even if not yet stable (handles sharp visual edges in blurry transitions)
            if color_diff > color_threshold * 2.5:
                first_stable_next = dist
                break

            # Check if color has stabilized (look ahead a bit)
            if dist + 2 <= max_dist:
                x2 = start_x + dx * (dist + 1)
                y2 = start_y + dy * (dist + 1)
                x3 = start_x + dx * (dist + 2)
                y3 = start_y + dy * (dist + 2)

                if (0 <= x2 < width and 0 <= y2 < height and
                    0 <= x3 < width and 0 <= y3 < height):
                    if len(img_array.shape) == 3:
                        c2 = img_array[y2, x2, :3].astype(float)
                        c3 = img_array[y3, x3, :3].astype(float)
                    else:
                        c2 = np.array([img_array[y2, x2]]).astype(float)
                        c3 = np.array([img_array[y3, x3]]).astype(float)

                    # Check if next pixels are similar to current (stable)
                    # AND significantly different from center (truly in neighbor region)
                    diff_to_next = np.sqrt(np.sum((pixel_color - c2) ** 2))
                    diff_next_pair = np.sqrt(np.sum((c2 - c3) ** 2))
                    diff_from_center = np.sqrt(np.sum((pixel_color - center_color) ** 2))

                    # Must be stable (consecutive pixels similar) AND
                    # clearly in neighbor territory (far from center color)
                    if (diff_to_next < color_threshold and
                        diff_next_pair < color_threshold and
                        diff_from_center > color_threshold * 1.5):
                        # Color has stabilized at this new value
                        first_stable_next = dist
                        break
            else:
                # Near max dist, assume stable if different
                first_stable_next = dist
                break

    if first_stable_next is None:
        # No edge found - same color continues
        return None, None

    return last_stable_mine, first_stable_next


def find_center_from_edges(
    img_array: np.ndarray,
    cx: int,
    cy: int,
    cell_w: float,
    cell_h: float,
    position_grid: dict | None = None,
    col: int = 0,
    row: int = 0,
) -> tuple[float, float, bool]:
    """
    Find center by locating edge midpoints in all 4 directions.
    Returns (new_cx, new_cy, is_confident) as floats.

    For edges that don't exist (same color neighbor), uses position_grid
    to infer the coordinate from nearby placed centers.
    """
    height, width = img_array.shape[:2]
    max_dist = int(max(cell_w, cell_h) * 0.8)

    # Get center color
    if len(img_array.shape) == 3:
        center_color = img_array[cy, cx, :3].astype(float)
    else:
        center_color = np.array([img_array[cy, cx]]).astype(float)

    # Find edges in all 4 directions
    left = find_edge_bounds(img_array, cx, cy, -1, 0, center_color, max_dist)
    right = find_edge_bounds(img_array, cx, cy, 1, 0, center_color, max_dist)
    up = find_edge_bounds(img_array, cx, cy, 0, -1, center_color, max_dist)
    down = find_edge_bounds(img_array, cx, cy, 0, 1, center_color, max_dist)

    new_cx, new_cy = float(cx), float(cy)
    confident = True

    # Horizontal center from left/right edges
    if left[0] is not None and right[0] is not None:
        # Both edges found - midpoint of edge midpoints
        left_edge = cx - (left[0] + left[1]) / 2
        right_edge = cx + (right[0] + right[1]) / 2
        new_cx = (left_edge + right_edge) / 2
    elif left[0] is not None:
        # Only left edge - use cell width to estimate
        left_edge = cx - (left[0] + left[1]) / 2
        new_cx = left_edge + cell_w / 2
        confident = False
    elif right[0] is not None:
        # Only right edge
        right_edge = cx + (right[0] + right[1]) / 2
        new_cx = right_edge - cell_w / 2
        confident = False
    elif position_grid is not None:
        # No edges - infer from neighbors
        for dc in [-1, 1]:
            nc = col + dc
            if (nc, row) in position_grid:
                neighbor_cx, _ = position_grid[(nc, row)]
                new_cx = neighbor_cx - dc * cell_w
                confident = False
                break

    # Vertical center from up/down edges
    if up[0] is not None and down[0] is not None:
        up_edge = cy - (up[0] + up[1]) / 2
        down_edge = cy + (down[0] + down[1]) / 2
        new_cy = (up_edge + down_edge) / 2
    elif up[0] is not None:
        up_edge = cy - (up[0] + up[1]) / 2
        new_cy = up_edge + cell_h / 2
        confident = False
    elif down[0] is not None:
        down_edge = cy + (down[0] + down[1]) / 2
        new_cy = down_edge - cell_h / 2
        confident = False
    elif position_grid is not None:
        for dr in [-1, 1]:
            nr = row + dr
            if (col, nr) in position_grid:
                _, neighbor_cy = position_grid[(col, nr)]
                new_cy = neighbor_cy - dr * cell_h
                confident = False
                break

    return new_cx, new_cy, confident


def compute_center_confidence(
    img_array: np.ndarray,
    cx: int,
    cy: int,
    cell_w: float,
    cell_h: float,
) -> float:
    """
    Compute confidence score for a center based on nearby edge strength.
    Higher = more confident (strong edges on ALL 4 sides).

    Returns the MINIMUM edge strength across all 4 boundaries.
    A pixel is only confident if all 4 edges are present.
    """
    height, width = img_array.shape[:2]

    half_w = int(cell_w / 2)
    half_h = int(cell_h / 2)

    edge_strengths = []

    # Left boundary - check horizontal gradient
    bx, by = cx - half_w, cy
    if 1 <= bx < width - 1 and 0 <= by < height:
        if len(img_array.shape) == 3:
            grad = np.sqrt(np.sum((img_array[by, bx+1, :3].astype(float) -
                                   img_array[by, bx-1, :3].astype(float)) ** 2))
        else:
            grad = abs(float(img_array[by, bx+1]) - float(img_array[by, bx-1]))
        edge_strengths.append(grad)

    # Right boundary - check horizontal gradient
    bx, by = cx + half_w, cy
    if 1 <= bx < width - 1 and 0 <= by < height:
        if len(img_array.shape) == 3:
            grad = np.sqrt(np.sum((img_array[by, bx+1, :3].astype(float) -
                                   img_array[by, bx-1, :3].astype(float)) ** 2))
        else:
            grad = abs(float(img_array[by, bx+1]) - float(img_array[by, bx-1]))
        edge_strengths.append(grad)

    # Top boundary - check vertical gradient
    bx, by = cx, cy - half_h
    if 0 <= bx < width and 1 <= by < height - 1:
        if len(img_array.shape) == 3:
            grad = np.sqrt(np.sum((img_array[by+1, bx, :3].astype(float) -
                                   img_array[by-1, bx, :3].astype(float)) ** 2))
        else:
            grad = abs(float(img_array[by+1, bx]) - float(img_array[by-1, bx]))
        edge_strengths.append(grad)

    # Bottom boundary - check vertical gradient
    bx, by = cx, cy + half_h
    if 0 <= bx < width and 1 <= by < height - 1:
        if len(img_array.shape) == 3:
            grad = np.sqrt(np.sum((img_array[by+1, bx, :3].astype(float) -
                                   img_array[by-1, bx, :3].astype(float)) ** 2))
        else:
            grad = abs(float(img_array[by+1, bx]) - float(img_array[by-1, bx]))
        edge_strengths.append(grad)

    # Return minimum - confident only if ALL edges are strong
    if len(edge_strengths) == 4:
        return min(edge_strengths)
    else:
        return 0.0  # Missing boundaries = not confident


def refine_centers_locally(
    img_array: np.ndarray,
    centers: list[tuple[int, int, int, int]],
    cell_w: float,
    cell_h: float,
    search_radius: float = 0.3,
    debug_path: str | Path | None = None,
    debug_img: Image.Image | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Refine each center locally to find the best position.
    Uses confidence weighting: high-confidence centers anchor low-confidence ones.

    Args:
        img_array: Input image
        centers: List of (col, row, cx, cy) tuples
        cell_w, cell_h: Cell dimensions
        search_radius: Fraction of cell size to search (0.3 = ±30%)

    Returns:
        Refined centers list
    """
    height, width = img_array.shape[:2]
    n_cols = max(c[0] for c in centers) + 1
    n_rows = max(c[1] for c in centers) + 1

    # Search window size
    search_w = max(2, int(cell_w * search_radius))
    search_h = max(2, int(cell_h * search_radius))

    # Scoring radius
    score_radius = max(2, int(min(cell_w, cell_h) * 0.25))

    # First pass: refine all centers using edge midpoints and compute confidence
    refined_with_conf = {}

    for col, row, cx, cy in centers:
        # Refine by finding midpoint between edge midpoints
        refined_cx, refined_cy, is_confident = find_center_from_edges(
            img_array, cx, cy, cell_w, cell_h
        )

        # Clamp refinement to max 2 pixels in each dimension
        dx = refined_cx - cx
        dy = refined_cy - cy
        clamped_dx = max(-2, min(2, dx))
        clamped_dy = max(-2, min(2, dy))
        best_cx = cx + clamped_dx
        best_cy = cy + clamped_dy

        # Reduce confidence if we had to clamp
        if abs(dx) > 2 or abs(dy) > 2:
            is_confident = False

        # Round only for confidence calculation (needs integer coords for array access)
        confidence = compute_center_confidence(img_array, int(round(best_cx)), int(round(best_cy)), cell_w, cell_h)
        # Reduce confidence if edge detection wasn't confident
        if not is_confident:
            confidence *= 0.5
        refined_with_conf[(col, row)] = (best_cx, best_cy, confidence)

    # Sort confidences to determine percentile thresholds
    all_conf = sorted([c[2] for c in refined_with_conf.values()], reverse=True)

    # Iteratively place centers from high confidence to low
    # Start with top 10%, then 20%, 30%, etc.
    position_grid = {}  # (col, row) -> (cx, cy) once placed

    percentile_steps = [98, 95, 90, 80, 60, 25, 0]

    for step_idx, percentile in enumerate(percentile_steps):
        conf_threshold = np.percentile(all_conf, percentile) if percentile > 0 else -1
        placed_this_round = 0

        for col, row, orig_cx, orig_cy in centers:
            if (col, row) in position_grid:
                continue  # already placed

            cx, cy, conf = refined_with_conf[(col, row)]

            if conf >= conf_threshold:
                # This cell qualifies at this confidence level
                # Check if we have already-placed neighbors to constrain us
                predictions = []

                for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nc, nr = col + dc, row + dr
                    if (nc, nr) in position_grid:
                        ncx, ncy = position_grid[(nc, nr)]
                        pred_cx = ncx - dc * cell_w
                        pred_cy = ncy - dr * cell_h
                        predictions.append((pred_cx, pred_cy))

                if predictions:
                    # Blend refined position with neighbor predictions
                    avg_pred_cx = np.mean([p[0] for p in predictions])
                    avg_pred_cy = np.mean([p[1] for p in predictions])

                    # Higher confidence = trust refined more, lower = trust neighbors more
                    # Normalize confidence to 0-1 range
                    max_conf = all_conf[0] if all_conf[0] > 0 else 1
                    norm_conf = min(1.0, conf / max_conf)

                    # Blend: high conf -> more weight on refined, low conf -> more weight on neighbors
                    blend = 0.3 + 0.5 * norm_conf  # ranges from 0.3 to 0.8
                    final_cx = cx * blend + avg_pred_cx * (1 - blend)
                    final_cy = cy * blend + avg_pred_cy * (1 - blend)

                    position_grid[(col, row)] = (final_cx, final_cy)
                else:
                    # No neighbors yet - use refined position directly
                    position_grid[(col, row)] = (cx, cy)

                placed_this_round += 1

        # Save debug image for this iteration
        if debug_path and debug_img:
            iter_path = Path(debug_path).parent / f"{Path(debug_path).stem}_iter{step_idx}_p{percentile}.png"
            current_centers = [(int(round(cx)), int(round(cy))) for cx, cy in position_grid.values()]
            save_debug_centers(debug_img, current_centers, iter_path)

            # Also save edge visualization for first iteration
            if step_idx == 0:
                edges_path = Path(debug_path).parent / f"{Path(debug_path).stem}_edges.png"
                placed_centers = [(col, row, int(round(cx)), int(round(cy)))
                                  for (col, row), (cx, cy) in position_grid.items()]
                save_debug_edges(debug_img, img_array, placed_centers, cell_w, cell_h, edges_path)

    # Any remaining cells (shouldn't happen, but safety)
    for col, row, orig_cx, orig_cy in centers:
        if (col, row) not in position_grid:
            position_grid[(col, row)] = (orig_cx, orig_cy)

    # Build final centers list (round only at the very end)
    final_centers = []
    for col, row, orig_cx, orig_cy in centers:
        cx, cy = position_grid[(col, row)]
        final_centers.append((col, row, int(round(cx)), int(round(cy))))

    return final_centers


def find_best_grid(img_array: np.ndarray, min_period: int = 3, max_period: int | None = None) -> tuple[int, int, float, float]:
    """
    Find the best grid by optimizing cell size AND offset.
    Returns (n_cols, n_rows, offset_x, offset_y).
    """
    height, width = img_array.shape[:2]

    # Default max_period to 1/10th of smallest dimension (pixel art cells shouldn't be huge)
    if max_period is None:
        max_period = max(20, min(width, height) // 10)

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


def save_debug_edges(
    img: Image.Image,
    img_array: np.ndarray,
    centers: list[tuple[int, int, int, int]],  # (col, row, cx, cy)
    cell_w: float,
    cell_h: float,
    output_path: str | Path,
) -> None:
    """Save image with detected edge midpoints drawn as purple lines."""
    from PIL import ImageDraw
    debug_img = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(debug_img)

    height, width = img_array.shape[:2]
    max_dist = int(max(cell_w, cell_h) * 0.8)

    for col, row, cx, cy in centers:
        # Get center color
        if len(img_array.shape) == 3:
            center_color = img_array[cy, cx, :3].astype(float)
        else:
            center_color = np.array([img_array[cy, cx]]).astype(float)

        # Find edges in all 4 directions
        left = find_edge_bounds(img_array, cx, cy, -1, 0, center_color, max_dist)
        right = find_edge_bounds(img_array, cx, cy, 1, 0, center_color, max_dist)
        up = find_edge_bounds(img_array, cx, cy, 0, -1, center_color, max_dist)
        down = find_edge_bounds(img_array, cx, cy, 0, 1, center_color, max_dist)

        purple = (180, 0, 255, 200)
        line_len = int(cell_h * 0.4)

        # Draw vertical lines at left/right edges
        if left[0] is not None:
            edge_x = cx - (left[0] + left[1]) / 2
            draw.line([(edge_x, cy - line_len), (edge_x, cy + line_len)], fill=purple, width=1)
        if right[0] is not None:
            edge_x = cx + (right[0] + right[1]) / 2
            draw.line([(edge_x, cy - line_len), (edge_x, cy + line_len)], fill=purple, width=1)

        # Draw horizontal lines at up/down edges
        line_len = int(cell_w * 0.4)
        if up[0] is not None:
            edge_y = cy - (up[0] + up[1]) / 2
            draw.line([(cx - line_len, edge_y), (cx + line_len, edge_y)], fill=purple, width=1)
        if down[0] is not None:
            edge_y = cy + (down[0] + down[1]) / 2
            draw.line([(cx - line_len, edge_y), (cx + line_len, edge_y)], fill=purple, width=1)

        # Draw center as red dot
        draw.ellipse([(cx - 1, cy - 1), (cx + 1, cy + 1)], fill=(255, 0, 0, 255))

    debug_img.save(output_path)


def extract_pixel_art(
    input_path: str | Path,
    output_path: str | Path | None = None,
    verbose: bool = True,
    debug: bool = False,
    pass0_only: bool = False,
    pass1_only: bool = False,
) -> Image.Image | None:
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

    # Compute initial grid centers
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            cx = int(offset_x + (col + 0.5) * cell_w)
            cy = int(offset_y + (row + 0.5) * cell_h)

            # Clamp to image bounds
            cx = max(0, min(img.size[0] - 1, cx))
            cy = max(0, min(img.size[1] - 1, cy))

            centers.append((col, row, cx, cy))

    # Refine centers locally to find optimal positions
    debug_path_for_refine = None
    if debug and output_path:
        debug_path_for_refine = Path(output_path).parent / f"{Path(output_path).stem}_pass0.png"
    centers = refine_centers_locally(
        img_array, centers, cell_w, cell_h,
        debug_path=debug_path_for_refine,
        debug_img=img if debug else None,
    )

    if verbose:
        # Count how many centers moved
        initial_centers = []
        for row in range(n_rows):
            for col in range(n_cols):
                cx = int(offset_x + (col + 0.5) * cell_w)
                cy = int(offset_y + (row + 0.5) * cell_h)
                cx = max(0, min(img.size[0] - 1, cx))
                cy = max(0, min(img.size[1] - 1, cy))
                initial_centers.append((cx, cy))

        moved = sum(1 for (_, _, cx, cy), (icx, icy) in zip(centers, initial_centers)
                    if cx != icx or cy != icy)
        print(f"Refined {moved}/{len(centers)} centers locally")

    # Save pass0: debug overlay showing detected grid centers
    if output_path:
        pass0_path = Path(output_path).parent / f"{Path(output_path).stem}_pass0.png"
        save_debug_centers(img, [(c[2], c[3]) for c in centers], pass0_path)
        if verbose:
            print(f"Pass 0 (grid centers) saved to: {pass0_path}")

    if pass0_only:
        if verbose:
            print("Stopping after pass0 (--pass0-only)")
        return None

    if out_width <= 0 or out_height <= 0:
        raise ValueError("Could not detect valid pixel grid.")

    if verbose:
        print(f"Output size: {out_width}x{out_height}")

    # Create output image
    output = Image.new("RGBA", (out_width, out_height))

    # Pass 1: Sample inner 50-80% of each cell and take average (ignoring outliers)
    # This gives us the "true" color for each pixel before palette reduction
    sampled_colors = []
    inner_fraction = 0.6  # Sample inner 60% of cell (30% margin on each side)
    inner_radius_w = max(1, int(cell_w * inner_fraction / 2))
    inner_radius_h = max(1, int(cell_h * inner_fraction / 2))

    for col, row, cx, cy in centers:
        # Clamp center to valid bounds
        cx = max(0, min(img.size[0] - 1, cx))
        cy = max(0, min(img.size[1] - 1, cy))

        # Sample inner region of the cell
        x1 = max(0, cx - inner_radius_w)
        x2 = min(img.size[0], cx + inner_radius_w + 1)
        y1 = max(0, cy - inner_radius_h)
        y2 = min(img.size[1], cy + inner_radius_h + 1)

        region = img_array[y1:y2, x1:x2]

        if region.size > 0 and len(region.shape) == 3:
            # Flatten to list of RGB values
            pixels = region.reshape(-1, region.shape[-1])[:, :3].astype(float)

            if len(pixels) >= 4:
                # Remove outliers using IQR on color distance from median
                median_color = np.median(pixels, axis=0)
                distances = np.sqrt(np.sum((pixels - median_color) ** 2, axis=1))
                q1, q3 = np.percentile(distances, [25, 75])
                iqr = q3 - q1
                mask = distances <= q3 + 1.5 * iqr
                filtered = pixels[mask] if np.sum(mask) > 0 else pixels

                # Take mean of filtered pixels
                mean_color = np.mean(filtered, axis=0)
            else:
                mean_color = np.mean(pixels, axis=0)

            color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]), 255)
        else:
            pixel = img_array[cy, cx]
            if len(pixel) >= 3:
                color = (int(pixel[0]), int(pixel[1]), int(pixel[2]), 255)
            else:
                color = (int(pixel[0]), int(pixel[0]), int(pixel[0]), 255)

        sampled_colors.append((col, row, color))

    # Create pass1 image (true colors before palette reduction)
    pass1_image = Image.new("RGBA", (out_width, out_height))
    for col, row, color in sampled_colors:
        pass1_image.putpixel((col, row), color)

    # Save pass1: true colors before palette reduction
    if output_path:
        pass1_path = Path(output_path).parent / f"{Path(output_path).stem}_pass1.png"
        pass1_image.save(pass1_path)
        if verbose:
            print(f"Pass 1 (true colors) saved to: {pass1_path}")

    if pass1_only:
        if verbose:
            print("Stopping after pass1 (--pass1-only)")
        return pass1_image

    # Spatial pre-clustering: chain similar adjacent pixels
    if verbose:
        print("Spatial clustering...")
    sampled_colors = precluster_colors_spatial(sampled_colors)
    if verbose:
        unique_after = len(set(c[2][:3] for c in sampled_colors))
        print(f"  {unique_after} unique colors after spatial clustering")

    # Extract palette using new quality metric
    all_colors = [c[2] for c in sampled_colors]
    palette = extract_palette_quality(all_colors, precluster_threshold=0, verbose=verbose)

    if verbose:
        print(f"Palette: {len(palette)} colors")
        # Show palette as hex colors
        hex_colors = ['#' + ''.join(f'{c:02x}' for c in p[:3]) for p in palette]
        print(f"  Colors: {', '.join(hex_colors)}")

    # Snap all colors to palette
    snapped_grid = {}
    for col, row, color in sampled_colors:
        snapped = snap_to_palette(color, palette)
        snapped_grid[(col, row)] = snapped

    # Clean up rare colors: if a palette color is used < N times,
    # replace with nearest other palette color if within threshold
    min_color_count = 3  # Colors used less than this many times
    rare_color_threshold = 25.0  # Only merge if nearest color is within this distance

    color_counts = {}
    for color in snapped_grid.values():
        color_counts[color] = color_counts.get(color, 0) + 1

    rare_colors = {c for c, count in color_counts.items() if count < min_color_count}

    if rare_colors and verbose:
        print(f"  Found {len(rare_colors)} rare colors (used <{min_color_count} times)")

    # Build mapping for rare colors to their nearest non-rare palette color
    rare_mapping = {}
    for rare in rare_colors:
        best_dist = float('inf')
        best_color = rare
        for other in palette:
            other_tuple = tuple(other)
            if other_tuple not in rare_colors and other_tuple != rare:
                dist = color_distance(rare[:3], other[:3])
                if dist < best_dist:
                    best_dist = dist
                    best_color = other_tuple
        if best_dist <= rare_color_threshold:
            rare_mapping[rare] = best_color

    if rare_mapping and verbose:
        print(f"  Merging {len(rare_mapping)} rare colors into nearby palette colors")

    # Apply rare color mapping
    for pos, color in snapped_grid.items():
        if color in rare_mapping:
            snapped_grid[pos] = rare_mapping[color]

    # Local smoothing: if a pixel has a very similar neighbor, use the more common color in 3x3 block
    similar_neighbor_threshold = 12.0  # Colors within this distance are "super close"
    smoothed = 0

    for (col, row), color in list(snapped_grid.items()):
        # Get 3x3 neighborhood colors
        neighbors = []
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                nc, nr = col + dc, row + dr
                if (nc, nr) in snapped_grid:
                    neighbors.append(snapped_grid[(nc, nr)])

        # Check if any neighbor is super close to this pixel's color
        has_similar = False
        for n in neighbors:
            if n != color and color_distance(color[:3], n[:3]) < similar_neighbor_threshold:
                has_similar = True
                break

        if has_similar:
            # Count colors in 3x3 block
            local_counts = {}
            for n in neighbors:
                local_counts[n] = local_counts.get(n, 0) + 1

            # Find the most common color that's similar to current
            best_color = color
            best_count = local_counts.get(color, 0)
            for c, count in local_counts.items():
                if count > best_count and color_distance(color[:3], c[:3]) < similar_neighbor_threshold:
                    best_color = c
                    best_count = count

            if best_color != color:
                snapped_grid[(col, row)] = best_color
                smoothed += 1

    if smoothed and verbose:
        print(f"  Smoothed {smoothed} pixels to match local neighbors")

    # Remove unused colors from palette
    used_colors = set(snapped_grid.values())
    palette = [c for c in palette if tuple(c) in used_colors]

    # Merge similar palette colors globally
    palette_merge_threshold = 15.0
    palette_list = [tuple(c) for c in palette]
    merged_palette_map = {}  # old color -> new color

    # Sort by usage count (most used first) to keep the dominant color
    palette_usage = {c: sum(1 for v in snapped_grid.values() if v == c) for c in palette_list}
    palette_list.sort(key=lambda c: palette_usage[c], reverse=True)

    final_palette = []
    for color in palette_list:
        # Check if this color should merge into an existing final palette color
        merged = False
        for existing in final_palette:
            if color_distance(color[:3], existing[:3]) < palette_merge_threshold:
                merged_palette_map[color] = existing
                merged = True
                break
        if not merged:
            final_palette.append(color)
            merged_palette_map[color] = color

    if len(final_palette) < len(palette_list) and verbose:
        print(f"  Merged {len(palette_list) - len(final_palette)} similar palette colors")

    # Re-snap pixels to merged palette
    for pos, color in snapped_grid.items():
        if color in merged_palette_map:
            snapped_grid[pos] = merged_palette_map[color]

    palette = list(final_palette)

    if verbose:
        print(f"  Final palette: {len(palette)} colors")

    # Write to output
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
    parser.add_argument("--pass0-only", action="store_true", help="Stop after pass0 (grid detection only)")
    parser.add_argument("--pass1-only", action="store_true", help="Stop after pass1 (true colors, no palette reduction)")

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
        pass0_only=args.pass0_only,
        pass1_only=args.pass1_only,
    )


if __name__ == "__main__":
    main()
