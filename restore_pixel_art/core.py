"""
Extract clean pixel art from AI-generated upscaled pixel art images.

The input is expected to be pixel art where each "logical pixel" is rendered
as a larger block with potentially soft/blurred edges. Cell sizes may vary
throughout the image. This script detects the actual grid lines adaptively,
extracts the dominant color from each cell, and outputs a clean 1:1 pixel art image.
"""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter


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

    # Collect scores for all periods, weighted by inverse period
    # This favors smaller periods (more likely to be the fundamental)
    max_p = min(max_period + 1, n // 2)
    # Weight = score / sqrt(period) - balances raw score with period preference
    scores = [(period, autocorr[period], autocorr[period] / np.sqrt(period))
              for period in range(min_period, max_p)]

    if not scores:
        return min_period

    # Find best by weighted score
    best_period, best_raw, best_weighted = max(scores, key=lambda x: x[2])

    # Also check: if a smaller period is within 20% of best weighted score
    # and is a divisor of the best, strongly prefer it
    for period, raw, weighted in scores:
        if period >= best_period:
            break
        if best_period % period == 0 and weighted >= best_weighted * 0.8:
            return period

    return best_period


def find_best_grid(img_array: np.ndarray, min_period: int = 3, max_period: int | None = None) -> tuple[int, int, float, float, float, float, int, int]:
    """
    Find the best grid by optimizing cell size AND offset.
    Returns (n_cols, n_rows, offset_x, offset_y, cell_w, cell_h, start_partial_x, start_partial_y).
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
    # When asymmetric, trust the larger value (smaller is likely noise/artifacts)
    if cell_w > 0 and cell_h > 0:
        ratio = max(cell_w, cell_h) / min(cell_w, cell_h)
        if ratio > 1.5:
            cell_size = max(cell_w, cell_h)
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

    # Count full cells plus partial cells at start/end if > half a cell
    n_full_cols = int((width - offset_x) / cell_w)
    n_full_rows = int((height - offset_y) / cell_h)

    # Partial cell at start (from 0 to offset)
    start_partial_x = 1 if offset_x > cell_w * 0.5 else 0
    start_partial_y = 1 if offset_y > cell_h * 0.5 else 0

    # Partial cell at end (remainder after last full cell)
    end_remainder_x = (width - offset_x) - n_full_cols * cell_w
    end_remainder_y = (height - offset_y) - n_full_rows * cell_h
    end_partial_x = 1 if end_remainder_x > cell_w * 0.5 else 0
    end_partial_y = 1 if end_remainder_y > cell_h * 0.5 else 0

    n_cols = start_partial_x + n_full_cols + end_partial_x
    n_rows = start_partial_y + n_full_rows + end_partial_y

    return max(1, n_cols), max(1, n_rows), offset_x, offset_y, cell_w, cell_h, start_partial_x, start_partial_y


def save_debug_centers(
    img: Image.Image,
    centers: list[tuple[int, int]],
    output_path: str | Path,
) -> None:
    """Save a copy of the image with detected pixel centers marked."""
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

    if verbose:
        print(f"Input image: {img.size[0]}x{img.size[1]}")

    # Auto-scale to get cell size into optimal range (8-16 pixels)
    min_cell_size = 8
    max_cell_size = 16
    scale_factor = 1

    # Quick grid detection to check cell size
    img_array = np.array(img)
    n_cols, n_rows, offset_x, offset_y, cell_w, cell_h, start_partial_x, start_partial_y = find_best_grid(img_array)
    avg_cell = (cell_w + cell_h) / 2

    while avg_cell < min_cell_size and scale_factor < 16:
        scale_factor *= 2
        new_size = (img.size[0] * scale_factor, img.size[1] * scale_factor)
        scaled_img = img.resize(new_size, Image.Resampling.NEAREST)
        img_array = np.array(scaled_img)
        n_cols, n_rows, offset_x, offset_y, cell_w, cell_h, start_partial_x, start_partial_y = find_best_grid(img_array)
        avg_cell = (cell_w + cell_h) / 2

    while avg_cell > max_cell_size and scale_factor > 0.125:
        scale_factor *= 0.5
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        scaled_img = img.resize(new_size, Image.Resampling.LANCZOS)
        img_array = np.array(scaled_img)
        n_cols, n_rows, offset_x, offset_y, cell_w, cell_h, start_partial_x, start_partial_y = find_best_grid(img_array)
        avg_cell = (cell_w + cell_h) / 2

    if scale_factor != 1:
        if verbose:
            print(f"Auto-scaled by {scale_factor}x to get cell size ~{avg_cell:.1f}px")
        img = scaled_img

    if verbose:
        print(f"Best grid: {n_cols}x{n_rows} (cell size ~{cell_w:.1f}x{cell_h:.1f}px, offset {offset_x:.1f},{offset_y:.1f})")

    # Debug: save initial regular grid overlay
    if debug:
        # Scale so each cell is ~16 pixels for visibility
        debug_scale = max(1, int(round(16 / min(cell_w, cell_h))))
        img_w, img_h = img.size
        scaled_debug = img.resize((img_w * debug_scale, img_h * debug_scale), Image.NEAREST)
        draw_debug = ImageDraw.Draw(scaled_debug)

        # Draw regular grid lines based on detected parameters
        line_color = (255, 255, 0, 200)  # Yellow

        # Vertical lines
        x = offset_x
        while x < img_w:
            x_scaled = int(x * debug_scale)
            draw_debug.line([(x_scaled, 0), (x_scaled, img_h * debug_scale - 1)], fill=line_color, width=1)
            x += cell_w

        # Horizontal lines
        y = offset_y
        while y < img_h:
            y_scaled = int(y * debug_scale)
            draw_debug.line([(0, y_scaled), (img_w * debug_scale - 1, y_scaled)], fill=line_color, width=1)
            y += cell_h

        # Handle negative offsets
        x = offset_x - cell_w
        while x >= 0:
            x_scaled = int(x * debug_scale)
            draw_debug.line([(x_scaled, 0), (x_scaled, img_h * debug_scale - 1)], fill=line_color, width=1)
            x -= cell_w
        y = offset_y - cell_h
        while y >= 0:
            y_scaled = int(y * debug_scale)
            draw_debug.line([(0, y_scaled), (img_w * debug_scale - 1, y_scaled)], fill=line_color, width=1)
            y -= cell_h

        initial_grid_path = Path(output_path).parent / f"{Path(output_path).stem}_initial_grid.png"
        scaled_debug.save(initial_grid_path)
        print(f"Initial regular grid saved to: {initial_grid_path}")

    # Work directly with the original image (no blur needed since we use regular grid)
    img_array = np.array(img)
    working_img = img

    out_width = n_cols
    out_height = n_rows

    # Compute initial grid centers
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            # Handle partial cells at start
            if start_partial_x and col == 0:
                cx = int(offset_x / 2)  # Center of partial cell at start
            else:
                # Adjust col index if there's a start partial
                adj_col = col - start_partial_x
                cx = int(offset_x + (adj_col + 0.5) * cell_w)

            if start_partial_y and row == 0:
                cy = int(offset_y / 2)  # Center of partial cell at start
            else:
                adj_row = row - start_partial_y
                cy = int(offset_y + (adj_row + 0.5) * cell_h)

            # Clamp to image bounds
            cx = max(0, min(working_img.size[0] - 1, cx))
            cy = max(0, min(working_img.size[1] - 1, cy))

            centers.append((col, row, cx, cy))

    # Use the initial regular grid directly (no anchor refinement)

    # Save pass0: debug overlay showing detected grid centers (on blurred image)
    if debug and output_path:
        pass0_path = Path(output_path).parent / f"{Path(output_path).stem}_pass0.png"
        save_debug_centers(working_img, [(c[2], c[3]) for c in centers], pass0_path)
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

    # Apply median filter to blurred image for color sampling (further reduces artifacts)
    # Use smaller filter to preserve edge colors better
    denoise_size = max(1, int(min(cell_w, cell_h) / 6) | 1)  # Must be odd
    if len(img_array.shape) == 3:
        denoised_array = np.stack([
            median_filter(img_array[:, :, c], size=denoise_size)
            for c in range(img_array.shape[2])
        ], axis=2)
    else:
        denoised_array = median_filter(img_array, size=denoise_size)

    if verbose:
        print(f"Applied median filter (size={denoise_size}) for color sampling")

    # Pass 1: Sample inner 50-80% of each cell and take average (ignoring outliers)
    # This gives us the "true" color for each pixel before palette reduction
    sampled_colors = []
    inner_fraction = 0.6  # Sample inner 60% of cell (30% margin on each side)
    inner_radius_w = max(1, int(cell_w * inner_fraction / 2))
    inner_radius_h = max(1, int(cell_h * inner_fraction / 2))

    for col, row, cx, cy in centers:
        # Clamp center to valid bounds
        cx = max(0, min(working_img.size[0] - 1, cx))
        cy = max(0, min(working_img.size[1] - 1, cy))

        # Sample inner region of the cell (from denoised image)
        x1 = max(0, cx - inner_radius_w)
        x2 = min(working_img.size[0], cx + inner_radius_w + 1)
        y1 = max(0, cy - inner_radius_h)
        y2 = min(working_img.size[1], cy + inner_radius_h + 1)

        region = denoised_array[y1:y2, x1:x2]

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
    if debug and output_path:
        pass1_path = Path(output_path).parent / f"{Path(output_path).stem}_pass1.png"
        pass1_image.save(pass1_path)
        if verbose:
            print(f"Pass 1 (true colors) saved to: {pass1_path}")

    if pass1_only:
        if verbose:
            print("Stopping after pass1 (--pass1-only)")
        return pass1_image

    # Store original sampled colors as reference for max_deviation checks
    original_colors = {(col, row): color for col, row, color in sampled_colors}
    max_deviation = 12.0  # Strict limit for common/important colors
    max_deviation_rare = 22.0  # Relaxed limit for rare colors (merge them away)

    # We'll compute the rarity threshold after snapping to palette
    rarity_threshold = 0  # Will be set later based on median usage

    def is_valid_color(pos: tuple[int, int], new_color: tuple, is_rare: bool = False) -> bool:
        """Check if new_color is within allowed deviation of original color at pos.
        Rare colors get more tolerance since they're likely noise and should merge."""
        orig = original_colors[pos]
        effective_max = max_deviation_rare if is_rare else max_deviation
        return color_distance(orig[:3], new_color[:3]) <= effective_max

    # Spatial pre-clustering: chain similar adjacent pixels
    if verbose:
        unique_before = len(set(c[2][:3] for c in sampled_colors))
        print(f"Spatial clustering ({unique_before} unique colors)...")
    sampled_colors = precluster_colors_spatial(sampled_colors)
    if verbose:
        unique_after = len(set(c[2][:3] for c in sampled_colors))
        print(f"  {unique_after} unique colors after spatial clustering")

    # Validate spatial clustering didn't exceed max_deviation (strict check)
    valid_sampled = []
    spatial_reverts = 0
    for col, row, color in sampled_colors:
        if is_valid_color((col, row), color, is_rare=False):
            valid_sampled.append((col, row, color))
        else:
            # Revert to original color
            valid_sampled.append((col, row, original_colors[(col, row)]))
            spatial_reverts += 1
    sampled_colors = valid_sampled
    if spatial_reverts and verbose:
        print(f"  Reverted {spatial_reverts} pixels that exceeded max_deviation")

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

    # Compute rarity threshold based on median palette usage / 4
    color_counts = {}
    for color in snapped_grid.values():
        color_counts[color] = color_counts.get(color, 0) + 1

    usage_values = sorted(color_counts.values())
    median_usage = usage_values[len(usage_values) // 2] if usage_values else 10
    rarity_threshold = max(3, median_usage // 4)  # At least 3, or median/4

    if verbose:
        print(f"  Rarity threshold: {rarity_threshold} (median usage: {median_usage})")

    # Clean up rare colors: if a palette color is used < rarity_threshold times,
    # replace with nearest other palette color (rare colors get relaxed deviation check)
    rare_color_merge_threshold = 25.0  # Only merge if nearest color is within this distance

    rare_colors = {c for c, count in color_counts.items() if count < rarity_threshold}

    if rare_colors and verbose:
        print(f"  Found {len(rare_colors)} rare colors (used <{rarity_threshold} times)")

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
        if best_dist <= rare_color_merge_threshold:
            rare_mapping[rare] = best_color

    if rare_mapping and verbose:
        print(f"  Merging {len(rare_mapping)} rare colors into nearby palette colors")

    # Apply rare color mapping (with relaxed max_deviation check for rare colors)
    rare_applied = 0
    rare_blocked = 0
    for pos, color in list(snapped_grid.items()):
        if color in rare_mapping:
            new_color = rare_mapping[color]
            # Rare colors get relaxed threshold - we WANT to merge them away
            if is_valid_color(pos, new_color, is_rare=True):
                snapped_grid[pos] = new_color
                rare_applied += 1
            else:
                rare_blocked += 1
    if rare_blocked and verbose:
        print(f"    Blocked {rare_blocked} merges (exceeded even relaxed threshold)")

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
                if is_valid_color((col, row), best_color, is_rare=False):
                    snapped_grid[(col, row)] = best_color
                    smoothed += 1

    if smoothed and verbose:
        print(f"  Smoothed {smoothed} pixels to match local neighbors")

    # Remove unused colors from palette
    used_colors = set(snapped_grid.values())
    palette = [c for c in palette if tuple(c) in used_colors]

    # Adjacency-aware merge: if a color ALWAYS appears next to a similar color,
    # it's likely an artifact from blending/compression, not an intentional color
    adjacency_merge_threshold = 18.0  # Conservative to preserve intentional color variations
    palette_list = [tuple(c) for c in palette]
    palette_usage = {c: sum(1 for v in snapped_grid.values() if v == c) for c in palette_list}

    # Build adjacency map and check for isolated pixels (no same-color in 8-neighbors)
    color_neighbors: dict[tuple, set[tuple]] = {c: set() for c in palette_list}
    isolated_pixels: dict[tuple, int] = {c: 0 for c in palette_list}  # count of isolated instances
    total_pixels: dict[tuple, int] = {c: 0 for c in palette_list}

    for (col, row), color in snapped_grid.items():
        total_pixels[color] += 1
        has_same_color_neighbor = False
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if dc == 0 and dr == 0:
                    continue
                nc, nr = col + dc, row + dr
                if (nc, nr) in snapped_grid:
                    neighbor_color = snapped_grid[(nc, nr)]
                    if neighbor_color == color:
                        has_same_color_neighbor = True
                    elif dc == 0 or dr == 0:  # Only 4-connected for different colors
                        color_neighbors[color].add(neighbor_color)
        if not has_same_color_neighbor:
            isolated_pixels[color] += 1

    # Find colors that should merge due to always being adjacent to a similar color
    adjacency_merges = {}
    for color in palette_list:
        if color in adjacency_merges:
            continue
        neighbors = color_neighbors[color]
        for neighbor in neighbors:
            if neighbor in adjacency_merges:
                continue
            dist = color_distance(color[:3], neighbor[:3])
            if dist < adjacency_merge_threshold:
                # Check if the rarer color ONLY appears next to the more common one
                rare_color = color if palette_usage[color] < palette_usage[neighbor] else neighbor
                common_color = neighbor if rare_color == color else color
                rare_neighbors = color_neighbors[rare_color]

                # Strong signal: color is mostly isolated single pixels
                if total_pixels[rare_color] > 0:
                    isolation_rate = isolated_pixels[rare_color] / total_pixels[rare_color]
                    if isolation_rate >= 0.7 and common_color in rare_neighbors:
                        # More than half are isolated - definitely artifact
                        adjacency_merges[rare_color] = common_color
                        continue

                # If the rare color's only neighbor (or primary neighbor) is the common similar color
                # then it's likely an artifact
                if len(rare_neighbors) == 1 and common_color in rare_neighbors:
                    adjacency_merges[rare_color] = common_color
                elif len(rare_neighbors) <= 3 and common_color in rare_neighbors:
                    # Even with a few neighbors, if most instances are next to the similar color, merge
                    # Count how many rare pixels are adjacent to common color
                    rare_positions = [(p, c) for p, c in snapped_grid.items() if c == rare_color]
                    adjacent_to_common = 0
                    for (col, row), _ in rare_positions:
                        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            if snapped_grid.get((col + dc, row + dr)) == common_color:
                                adjacent_to_common += 1
                                break
                    if adjacent_to_common >= len(rare_positions) * 0.8:  # 80% adjacent
                        adjacency_merges[rare_color] = common_color

    if adjacency_merges and verbose:
        print(f"  Merged {len(adjacency_merges)} artifact colors (isolated/always adjacent to similar)")

    # Apply adjacency merges (with max_deviation check)
    # Colors being merged away that are rare get relaxed threshold
    adj_applied = 0
    adj_blocked = 0
    for pos, color in list(snapped_grid.items()):
        if color in adjacency_merges:
            new_color = adjacency_merges[color]
            pixel_count = total_pixels.get(color, 1)
            is_rare = pixel_count < rarity_threshold
            if is_valid_color(pos, new_color, is_rare=is_rare):
                snapped_grid[pos] = new_color
                adj_applied += 1
            else:
                adj_blocked += 1
    if adj_blocked and verbose:
        print(f"    Blocked {adj_blocked} merges that would exceed max_deviation")

    # Update palette
    used_colors = set(snapped_grid.values())
    palette = [c for c in palette if tuple(c) in used_colors]

    # Second pass: clean up any newly-rare colors after adjacency merges
    color_counts = {}
    for color in snapped_grid.values():
        color_counts[color] = color_counts.get(color, 0) + 1

    rare_colors = {c for c, count in color_counts.items() if count < rarity_threshold}
    if rare_colors:
        rare_mapping = {}
        for rare in rare_colors:
            best_dist = float('inf')
            best_color = None
            for other in palette:
                other_tuple = tuple(other)
                if other_tuple not in rare_colors and other_tuple != rare:
                    dist = color_distance(rare[:3], other_tuple[:3])
                    if dist < best_dist:
                        best_dist = dist
                        best_color = other_tuple
            # Only merge if within threshold
            if best_color and best_dist <= rare_color_merge_threshold:
                rare_mapping[rare] = best_color

        if rare_mapping:
            cleanup_applied = 0
            cleanup_blocked = 0
            for pos, color in list(snapped_grid.items()):
                if color in rare_mapping:
                    new_color = rare_mapping[color]
                    # These are rare colors, use relaxed threshold
                    if is_valid_color(pos, new_color, is_rare=True):
                        snapped_grid[pos] = new_color
                        cleanup_applied += 1
                    else:
                        cleanup_blocked += 1
            if verbose:
                print(f"  Cleaned up {cleanup_applied} newly-rare colors after artifact merge")
                if cleanup_blocked:
                    print(f"    Blocked {cleanup_blocked} cleanups (exceeded even relaxed threshold)")

        # Update palette again
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

    # Re-snap pixels to merged palette (with max_deviation check)
    # Rare colors get relaxed threshold
    merge_applied = 0
    merge_blocked = 0
    for pos, color in list(snapped_grid.items()):
        if color in merged_palette_map:
            new_color = merged_palette_map[color]
            if new_color != color:  # Only check if actually changing
                is_rare = palette_usage.get(color, 0) < rarity_threshold
                if is_valid_color(pos, new_color, is_rare=is_rare):
                    snapped_grid[pos] = new_color
                    merge_applied += 1
                else:
                    merge_blocked += 1
    if merge_blocked and verbose:
        print(f"    Blocked {merge_blocked} palette merges that would exceed max_deviation")

    palette = list(final_palette)

    # Add back any colors that couldn't be merged due to max_deviation
    used_colors = set(snapped_grid.values())
    for color in used_colors:
        if color not in [tuple(p) for p in palette]:
            palette.append(color)

    # Final palette optimization: adjust each palette color to be the centroid
    # of the original (true) colors of all pixels assigned to it
    # This minimizes average error for each palette color
    if verbose:
        print("  Optimizing palette colors to minimize error...")

    # Group pixels by their current palette color
    color_to_positions: dict[tuple, list[tuple]] = {}
    for pos, color in snapped_grid.items():
        if color not in color_to_positions:
            color_to_positions[color] = []
        color_to_positions[color].append(pos)

    # Compute optimal color for each palette entry (average of original colors)
    optimized_palette = []
    color_remap = {}  # old palette color -> new optimized color

    for palette_color in palette:
        palette_tuple = tuple(palette_color) if not isinstance(palette_color, tuple) else palette_color
        positions = color_to_positions.get(palette_tuple, [])

        if not positions:
            optimized_palette.append(palette_color)
            color_remap[palette_tuple] = palette_tuple
            continue

        # Get original colors for all pixels with this palette color
        orig_colors = [original_colors[pos] for pos in positions]

        # Compute average
        avg_r = sum(c[0] for c in orig_colors) // len(orig_colors)
        avg_g = sum(c[1] for c in orig_colors) // len(orig_colors)
        avg_b = sum(c[2] for c in orig_colors) // len(orig_colors)
        optimal_color = (avg_r, avg_g, avg_b, 255)

        # Verify this doesn't exceed max_deviation for any pixel
        all_valid = all(
            color_distance(original_colors[pos][:3], optimal_color[:3]) <= max_deviation
            for pos in positions
        )

        if all_valid:
            optimized_palette.append(optimal_color)
            color_remap[palette_tuple] = optimal_color
        else:
            # Keep original palette color
            optimized_palette.append(palette_color)
            color_remap[palette_tuple] = palette_tuple

    # Apply optimized colors
    for pos, color in list(snapped_grid.items()):
        if color in color_remap:
            snapped_grid[pos] = color_remap[color]

    palette = optimized_palette

    # Final pass: re-snap each pixel to the closest palette color based on its
    # original (pass1) sampled color. This ensures we're using the best match
    # after all palette optimization/merging.
    resnapped = 0
    palette_tuples = [tuple(c) if not isinstance(c, tuple) else c for c in palette]
    for pos in snapped_grid:
        orig_color = original_colors[pos]
        current = snapped_grid[pos]
        # Find closest palette color to original
        best_color = current
        best_dist = color_distance(orig_color[:3], current[:3])
        for p in palette_tuples:
            dist = color_distance(orig_color[:3], p[:3])
            if dist < best_dist:
                best_dist = dist
                best_color = p
        if best_color != current:
            snapped_grid[pos] = best_color
            resnapped += 1

    if resnapped and verbose:
        print(f"  Re-snapped {resnapped} pixels to closest palette color")

    # Update palette to only include used colors
    used_colors = set(snapped_grid.values())
    palette = [c for c in palette if tuple(c) in used_colors]

    if verbose:
        hex_colors = ['#' + ''.join(f'{c:02x}' for c in p[:3]) for p in palette]
        print(f"  Final palette: {len(palette)} colors: {', '.join(hex_colors)}")

    # Write to output
    for (col, row), snapped in snapped_grid.items():
        output.putpixel((col, row), snapped)

    # Save if output path provided
    if output_path:
        output.save(output_path)
        if verbose:
            print(f"Saved to: {output_path}")

    return output

