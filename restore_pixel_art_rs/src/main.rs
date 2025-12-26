use clap::Parser;
use image::{ImageBuffer, Rgba, RgbaImage};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "restore-pixel-art")]
#[command(about = "Extract clean pixel art from AI-generated upscaled images")]
struct Args {
    /// Input image path
    input: PathBuf,

    /// Output image path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Suppress output
    #[arg(short, long)]
    quiet: bool,
}

fn apply_median_filter(img: &RgbaImage, size: usize) -> RgbaImage {
    if size <= 1 {
        return img.clone();
    }

    let (width, height) = img.dimensions();
    let mut result = img.clone();
    let half = (size / 2) as i32;

    for y in 0..height {
        for x in 0..width {
            let mut r_vals = Vec::with_capacity(size * size);
            let mut g_vals = Vec::with_capacity(size * size);
            let mut b_vals = Vec::with_capacity(size * size);

            for dy in -half..=half {
                for dx in -half..=half {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    let p = img.get_pixel(nx, ny);
                    r_vals.push(p[0]);
                    g_vals.push(p[1]);
                    b_vals.push(p[2]);
                }
            }

            r_vals.sort();
            g_vals.sort();
            b_vals.sort();

            let mid = r_vals.len() / 2;
            result.put_pixel(x, y, image::Rgba([r_vals[mid], g_vals[mid], b_vals[mid], img.get_pixel(x, y)[3]]));
        }
    }

    result
}

fn color_distance(c1: [u8; 3], c2: [u8; 3]) -> f64 {
    let dr = c1[0] as f64 - c2[0] as f64;
    let dg = c1[1] as f64 - c2[1] as f64;
    let db = c1[2] as f64 - c2[2] as f64;
    (dr * dr + dg * dg + db * db).sqrt()
}

fn compute_edge_signal(img: &RgbaImage, axis: usize) -> Vec<f64> {
    let (width, height) = img.dimensions();

    if axis == 0 {
        let mut signal = vec![0.0; height as usize - 1];
        for y in 0..(height - 1) {
            let mut sum = 0.0;
            for x in 0..width {
                let p1 = img.get_pixel(x, y);
                let p2 = img.get_pixel(x, y + 1);
                let gray1 = (p1[0] as f64 + p1[1] as f64 + p1[2] as f64) / 3.0;
                let gray2 = (p2[0] as f64 + p2[1] as f64 + p2[2] as f64) / 3.0;
                sum += (gray1 - gray2).abs();
            }
            signal[y as usize] = sum;
        }
        signal
    } else {
        let mut signal = vec![0.0; width as usize - 1];
        for x in 0..(width - 1) {
            let mut sum = 0.0;
            for y in 0..height {
                let p1 = img.get_pixel(x, y);
                let p2 = img.get_pixel(x + 1, y);
                let gray1 = (p1[0] as f64 + p1[1] as f64 + p1[2] as f64) / 3.0;
                let gray2 = (p2[0] as f64 + p2[1] as f64 + p2[2] as f64) / 3.0;
                sum += (gray1 - gray2).abs();
            }
            signal[x as usize] = sum;
        }
        signal
    }
}

fn find_period_autocorr(signal: &[f64], min_period: usize, max_period: usize) -> usize {
    let n = signal.len();
    if n == 0 {
        return min_period;
    }

    let mean: f64 = signal.iter().sum::<f64>() / n as f64;
    let std: f64 = (signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();

    let normalized: Vec<f64> = if std > 0.0 {
        signal.iter().map(|x| (x - mean) / std).collect()
    } else {
        signal.iter().map(|x| x - mean).collect()
    };

    // Collect scores for all periods, weighted by inverse sqrt(period)
    // This favors smaller periods (more likely to be the fundamental)
    let max_p = max_period.min(n / 2);
    let mut scores: Vec<(usize, f64, f64)> = Vec::new(); // (period, raw_score, weighted_score)

    for period in min_period..=max_p {
        let mut score = 0.0;
        for i in 0..(n - period) {
            score += normalized[i] * normalized[i + period];
        }
        let weighted = score / (period as f64).sqrt();
        scores.push((period, score, weighted));
    }

    if scores.is_empty() {
        return min_period;
    }

    // Find best by weighted score
    let (best_period, _best_raw, best_weighted) = scores
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .copied()
        .unwrap();

    // Check if a smaller period is a divisor and has good weighted score
    for &(period, _raw, weighted) in &scores {
        if period >= best_period {
            break;
        }
        if best_period % period == 0 && weighted >= best_weighted * 0.8 {
            return period;
        }
    }

    best_period
}

fn score_grid_alignment(edge_signal: &[f64], cell_size: f64, offset: f64) -> f64 {
    let length = edge_signal.len();
    let n_cells = ((length as f64 - offset) / cell_size) as usize;
    if n_cells < 2 {
        return 0.0;
    }

    let mut boundary_score = 0.0;
    let mut center_score = 0.0;

    for i in 1..n_cells {
        let boundary = (offset + i as f64 * cell_size) as usize;
        if boundary < length {
            boundary_score += edge_signal[boundary];
        }

        let center = (offset + (i as f64 - 0.5) * cell_size) as usize;
        if center < length {
            center_score += edge_signal[center];
        }
    }

    boundary_score - center_score * 0.5
}

struct GridResult {
    n_cols: usize,
    n_rows: usize,
    offset_x: f64,
    offset_y: f64,
    cell_w: f64,
    cell_h: f64,
    start_partial_x: bool,
    start_partial_y: bool,
}

fn find_best_grid(img: &RgbaImage) -> GridResult {
    let (width, height) = img.dimensions();
    let min_period = 3;
    let max_period = (width.min(height) / 10).max(20) as usize;

    let h_edges = compute_edge_signal(img, 0);
    let v_edges = compute_edge_signal(img, 1);

    let mut cell_w = find_period_autocorr(&v_edges, min_period, max_period) as f64;
    let mut cell_h = find_period_autocorr(&h_edges, min_period, max_period) as f64;

    if cell_w > 0.0 && cell_h > 0.0 {
        let ratio = cell_w.max(cell_h) / cell_w.min(cell_h);
        if ratio > 1.5 {
            let cell_size = cell_w.max(cell_h);
            cell_w = cell_size;
            cell_h = cell_size;
        }
    }

    let mut best_score_x = f64::NEG_INFINITY;
    let mut best_cw = cell_w;
    let mut best_ox = 0.0;

    for i in 0..31 {
        let cw = cell_w * (0.85 + 0.3 * i as f64 / 30.0);
        for j in 0..21 {
            // Match Python: np.linspace(0, cw - 0.1, 21)
            let ox = (cw - 0.1) * j as f64 / 20.0;
            let score = score_grid_alignment(&v_edges, cw, ox);
            if score > best_score_x {
                best_score_x = score;
                best_cw = cw;
                best_ox = ox;
            }
        }
    }

    let mut best_score_y = f64::NEG_INFINITY;
    let mut best_ch = cell_h;
    let mut best_oy = 0.0;

    for i in 0..31 {
        let ch = cell_h * (0.85 + 0.3 * i as f64 / 30.0);
        for j in 0..21 {
            // Match Python: np.linspace(0, ch - 0.1, 21)
            let oy = (ch - 0.1) * j as f64 / 20.0;
            let score = score_grid_alignment(&h_edges, ch, oy);
            if score > best_score_y {
                best_score_y = score;
                best_ch = ch;
                best_oy = oy;
            }
        }
    }

    cell_w = best_cw;
    cell_h = best_ch;
    let offset_x = best_ox;
    let offset_y = best_oy;

    let n_full_cols = ((width as f64 - offset_x) / cell_w) as usize;
    let n_full_rows = ((height as f64 - offset_y) / cell_h) as usize;

    let start_partial_x = offset_x > cell_w * 0.5;
    let start_partial_y = offset_y > cell_h * 0.5;

    let end_remainder_x = (width as f64 - offset_x) - n_full_cols as f64 * cell_w;
    let end_remainder_y = (height as f64 - offset_y) - n_full_rows as f64 * cell_h;
    let end_partial_x = end_remainder_x > cell_w * 0.5;
    let end_partial_y = end_remainder_y > cell_h * 0.5;

    let n_cols =
        (if start_partial_x { 1 } else { 0 }) + n_full_cols + (if end_partial_x { 1 } else { 0 });
    let n_rows =
        (if start_partial_y { 1 } else { 0 }) + n_full_rows + (if end_partial_y { 1 } else { 0 });

    GridResult {
        n_cols: n_cols.max(1),
        n_rows: n_rows.max(1),
        offset_x,
        offset_y,
        cell_w,
        cell_h,
        start_partial_x,
        start_partial_y,
    }
}

fn sample_color(img: &RgbaImage, cx: u32, cy: u32, cell_w: f64, cell_h: f64) -> [u8; 4] {
    let (width, height) = img.dimensions();
    let inner_w = (cell_w * 0.6 / 2.0).max(1.0) as i32;
    let inner_h = (cell_h * 0.6 / 2.0).max(1.0) as i32;

    // Collect all pixels in the inner region
    let mut pixels: Vec<[f64; 3]> = Vec::new();

    for dy in -inner_h..=inner_h {
        for dx in -inner_w..=inner_w {
            let x = (cx as i32 + dx).clamp(0, width as i32 - 1) as u32;
            let y = (cy as i32 + dy).clamp(0, height as i32 - 1) as u32;
            let p = img.get_pixel(x, y);
            pixels.push([p[0] as f64, p[1] as f64, p[2] as f64]);
        }
    }

    if pixels.len() >= 4 {
        // Remove outliers using IQR on color distance from median (match Python)
        // First compute median color
        let n = pixels.len();
        let mut r_vals: Vec<f64> = pixels.iter().map(|p| p[0]).collect();
        let mut g_vals: Vec<f64> = pixels.iter().map(|p| p[1]).collect();
        let mut b_vals: Vec<f64> = pixels.iter().map(|p| p[2]).collect();
        r_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        g_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        b_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_r = if n % 2 == 0 { (r_vals[n/2 - 1] + r_vals[n/2]) / 2.0 } else { r_vals[n/2] };
        let median_g = if n % 2 == 0 { (g_vals[n/2 - 1] + g_vals[n/2]) / 2.0 } else { g_vals[n/2] };
        let median_b = if n % 2 == 0 { (b_vals[n/2 - 1] + b_vals[n/2]) / 2.0 } else { b_vals[n/2] };

        // Compute distances from median
        let distances: Vec<f64> = pixels.iter()
            .map(|p| ((p[0] - median_r).powi(2) + (p[1] - median_g).powi(2) + (p[2] - median_b).powi(2)).sqrt())
            .collect();

        // Compute Q1, Q3, IQR
        let mut sorted_dist = distances.clone();
        sorted_dist.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;
        let q1 = sorted_dist[q1_idx];
        let q3 = sorted_dist[q3_idx];
        let iqr = q3 - q1;
        let threshold = q3 + 1.5 * iqr;

        // Filter outliers
        let filtered: Vec<&[f64; 3]> = pixels.iter()
            .zip(distances.iter())
            .filter(|&(_, d)| *d <= threshold)
            .map(|(p, _)| p)
            .collect();

        let filtered = if filtered.is_empty() { pixels.iter().collect::<Vec<_>>() } else { filtered };

        // Take mean of filtered pixels
        let sum_r: f64 = filtered.iter().map(|p| p[0]).sum();
        let sum_g: f64 = filtered.iter().map(|p| p[1]).sum();
        let sum_b: f64 = filtered.iter().map(|p| p[2]).sum();
        let count = filtered.len() as f64;

        [
            (sum_r / count) as u8,
            (sum_g / count) as u8,
            (sum_b / count) as u8,
            255,
        ]
    } else {
        // Simple average for small samples
        let sum_r: f64 = pixels.iter().map(|p| p[0]).sum();
        let sum_g: f64 = pixels.iter().map(|p| p[1]).sum();
        let sum_b: f64 = pixels.iter().map(|p| p[2]).sum();
        let count = pixels.len() as f64;

        [
            (sum_r / count) as u8,
            (sum_g / count) as u8,
            (sum_b / count) as u8,
            255,
        ]
    }
}

fn precluster_colors_spatial(
    colors: &mut [(u32, u32, [u8; 4])],
    neighbor_threshold: f64,
    chain_threshold: f64,
) {
    let mut grid: HashMap<(u32, u32), [u8; 4]> = HashMap::new();
    let mut max_col = 0u32;
    let mut max_row = 0u32;

    for &(col, row, color) in colors.iter() {
        grid.insert((col, row), color);
        max_col = max_col.max(col);
        max_row = max_row.max(row);
    }

    let mut chain_id: HashMap<(u32, u32), usize> = HashMap::new();
    let mut chains: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut chain_colors: Vec<[f64; 3]> = Vec::new();

    for row in 0..=max_row {
        for col in 0..=max_col {
            if let Some(&color) = grid.get(&(col, row)) {
                let rgb = [color[0] as f64, color[1] as f64, color[2] as f64];

                let mut best_chain: Option<usize> = None;
                let mut best_dist = f64::INFINITY;

                for (dc, dr) in [
                    (-1i32, -1i32),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ] {
                    let nc = col as i32 + dc;
                    let nr = row as i32 + dr;
                    if nc >= 0 && nr >= 0 {
                        if let Some(&cid) = chain_id.get(&(nc as u32, nr as u32)) {
                            if let Some(&neighbor_color) = grid.get(&(nc as u32, nr as u32)) {
                                let neighbor_rgb =
                                    [neighbor_color[0], neighbor_color[1], neighbor_color[2]];
                                let chain_avg = chain_colors[cid];

                                let neighbor_dist = color_distance(
                                    [rgb[0] as u8, rgb[1] as u8, rgb[2] as u8],
                                    neighbor_rgb,
                                );
                                let chain_dist = color_distance(
                                    [rgb[0] as u8, rgb[1] as u8, rgb[2] as u8],
                                    [chain_avg[0] as u8, chain_avg[1] as u8, chain_avg[2] as u8],
                                );

                                if neighbor_dist < neighbor_threshold
                                    && chain_dist < chain_threshold
                                {
                                    if neighbor_dist < best_dist {
                                        best_dist = neighbor_dist;
                                        best_chain = Some(cid);
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(cid) = best_chain {
                    chain_id.insert((col, row), cid);
                    chains[cid].push((col, row));
                    // Recalculate average from all original colors (match Python)
                    let mut sum_r = 0i32;
                    let mut sum_g = 0i32;
                    let mut sum_b = 0i32;
                    for &pos in &chains[cid] {
                        let c = grid[&pos];
                        sum_r += c[0] as i32;
                        sum_g += c[1] as i32;
                        sum_b += c[2] as i32;
                    }
                    let n = chains[cid].len() as i32;
                    chain_colors[cid] = [(sum_r / n) as f64, (sum_g / n) as f64, (sum_b / n) as f64];
                } else {
                    let new_id = chains.len();
                    chain_id.insert((col, row), new_id);
                    chains.push(vec![(col, row)]);
                    chain_colors.push(rgb);
                }
            }
        }
    }

    // Merge chains with similar colors (match Python)
    let merge_threshold = chain_threshold * 0.8;
    let mut chain_map: Vec<usize> = (0..chains.len()).collect();

    fn find_canonical(chain_map: &mut Vec<usize>, mut i: usize) -> usize {
        while chain_map[i] != i {
            i = chain_map[i];
        }
        i
    }

    for i in 0..chains.len() {
        for j in (i + 1)..chains.len() {
            let ci = find_canonical(&mut chain_map, i);
            let cj = find_canonical(&mut chain_map, j);
            if ci != cj {
                let dist = color_distance(
                    [chain_colors[ci][0] as u8, chain_colors[ci][1] as u8, chain_colors[ci][2] as u8],
                    [chain_colors[cj][0] as u8, chain_colors[cj][1] as u8, chain_colors[cj][2] as u8],
                );
                if dist < merge_threshold {
                    chain_map[cj] = ci;
                }
            }
        }
    }

    // Compute merged colors (average of merged chains weighted by chain size)
    // Use integer division to match Python
    let mut merged_colors: HashMap<usize, [i32; 3]> = HashMap::new();
    for i in 0..chains.len() {
        let ci = find_canonical(&mut chain_map, i);
        if !merged_colors.contains_key(&ci) {
            // Find all chains that merge to this one
            let member_chains: Vec<usize> = (0..chains.len())
                .filter(|&j| find_canonical(&mut chain_map, j) == ci)
                .collect();

            let total_pixels: i32 = member_chains.iter().map(|&j| chains[j].len() as i32).sum();
            let r: i32 = member_chains.iter().map(|&j| chain_colors[j][0] as i32 * chains[j].len() as i32).sum::<i32>() / total_pixels;
            let g: i32 = member_chains.iter().map(|&j| chain_colors[j][1] as i32 * chains[j].len() as i32).sum::<i32>() / total_pixels;
            let b: i32 = member_chains.iter().map(|&j| chain_colors[j][2] as i32 * chains[j].len() as i32).sum::<i32>() / total_pixels;
            merged_colors.insert(ci, [r, g, b]);
        }
    }

    for (col, row, color) in colors.iter_mut() {
        if let Some(&cid) = chain_id.get(&(*col, *row)) {
            let canonical = find_canonical(&mut chain_map, cid);
            let avg = merged_colors[&canonical];
            *color = [avg[0] as u8, avg[1] as u8, avg[2] as u8, 255];
        }
    }
}

fn snap_to_palette(color: [u8; 4], palette: &[[u8; 4]]) -> [u8; 4] {
    let mut best_dist = f64::INFINITY;
    let mut best_color = color;

    for &p in palette {
        let dist = color_distance([color[0], color[1], color[2]], [p[0], p[1], p[2]]);
        if dist < best_dist {
            best_dist = dist;
            best_color = [p[0], p[1], p[2], color[3]];
        }
    }

    best_color
}

fn quantize_to_n_colors(colors: &[[u8; 3]], n: usize) -> Vec<[u8; 3]> {
    if colors.len() <= n {
        return colors.to_vec();
    }

    let unique: Vec<[u8; 3]> = colors.iter().copied().collect::<HashSet<_>>().into_iter().collect();

    // Cluster similar colors
    let cluster_threshold = 20.0;
    let mut clusters: Vec<([u8; 3], usize, Vec<[u8; 3]>)> = Vec::new();

    for &color in colors {
        let mut found = false;
        for (rep, count, members) in &mut clusters {
            if color_distance(color, *rep) < cluster_threshold {
                *count += 1;
                members.push(color);
                found = true;
                break;
            }
        }
        if !found {
            clusters.push((color, 1, vec![color]));
        }
    }

    clusters.sort_by(|a, b| b.1.cmp(&a.1));

    // Pick most common color from each cluster
    let mut cluster_reps: Vec<([u8; 3], usize)> = Vec::new();
    for (_, count, members) in &clusters {
        let mut color_counts: HashMap<[u8; 3], usize> = HashMap::new();
        for &c in members {
            *color_counts.entry(c).or_insert(0) += 1;
        }
        let most_common = color_counts.into_iter().max_by_key(|&(_, c)| c).unwrap().0;
        cluster_reps.push((most_common, *count));
    }

    // Greedy selection
    let mut palette = vec![cluster_reps[0].0];

    for &(color, _) in &cluster_reps[1..] {
        if palette.len() >= n {
            break;
        }
        let min_dist = palette
            .iter()
            .map(|&p| color_distance(color, p))
            .fold(f64::INFINITY, f64::min);
        if min_dist > 15.0 {
            palette.push(color);
        }
    }

    // Fill gaps
    if palette.len() < n {
        for &c in &unique {
            if palette.len() >= n {
                break;
            }
            let err = palette
                .iter()
                .map(|&p| color_distance(c, p))
                .fold(f64::INFINITY, f64::min);
            if err > 10.0 && !palette.contains(&c) {
                palette.push(c);
            }
        }
    }

    palette
}

fn compute_palette_quality(
    colors: &[[u8; 3]],
    palette: &[[u8; 3]],
    soft_threshold: f64,
) -> (f64, f64) {
    if colors.is_empty() || palette.is_empty() {
        return (1.0, f64::INFINITY);
    }

    let mut deviating = 0;
    let mut max_err = 0.0f64;

    for &c in colors {
        let min_dist = palette
            .iter()
            .map(|&p| color_distance(c, p))
            .fold(f64::INFINITY, f64::min);
        if min_dist > soft_threshold {
            deviating += 1;
        }
        max_err = max_err.max(min_dist);
    }

    (deviating as f64 / colors.len() as f64, max_err)
}

fn extract_palette_quality(colors: &[[u8; 4]], verbose: bool) -> Vec<[u8; 4]> {
    let max_deviation = 12.0;
    let soft_threshold = 5.0;
    let max_deviation_rate = 0.05;

    let rgb_colors: Vec<[u8; 3]> = colors.iter().map(|c| [c[0], c[1], c[2]]).collect();
    let unique: HashSet<[u8; 3]> = rgb_colors.iter().copied().collect();

    if verbose {
        println!("  {} unique colors from {} samples", unique.len(), colors.len());
    }

    if unique.len() <= 2 {
        return unique.into_iter().map(|c| [c[0], c[1], c[2], 255]).collect();
    }

    let min_size = 10;
    let max_size = 80.min(unique.len());

    // Convert to Vec preserving insertion order (match Python's set behavior in 3.7+)
    // Python's set iteration is insertion-order, so we must NOT sort
    let mut unique_vec: Vec<[u8; 3]> = Vec::new();
    let mut seen: HashSet<[u8; 3]> = HashSet::new();
    for &c in &rgb_colors {
        if seen.insert(c) {
            unique_vec.push(c);
        }
    }

    let mut all_results: Vec<(usize, Vec<[u8; 3]>, f64, f64)> = Vec::new();

    for n in min_size..=max_size {
        let palette = quantize_to_n_colors(&unique_vec, n);
        let (dev_rate, max_err) = compute_palette_quality(&rgb_colors, &palette, soft_threshold);
        all_results.push((n, palette, dev_rate, max_err));

        if verbose {
            let passes = max_err <= max_deviation && dev_rate <= max_deviation_rate;
            let marker = if passes { " ✓" } else { "" };
            println!(
                "  {} colors: max_err={:.1}, dev_rate={:.1}%{}",
                n,
                max_err,
                dev_rate * 100.0,
                marker
            );
        }
    }

    // Pick smallest palette that meets both thresholds
    let valid: Vec<_> = all_results
        .iter()
        .filter(|(_, _, dr, e)| *e <= max_deviation && *dr <= max_deviation_rate)
        .collect();

    let best = if !valid.is_empty() {
        valid.iter().min_by_key(|(n, _, _, _)| n).unwrap()
    } else {
        // Try just max_err
        let valid_max: Vec<_> = all_results
            .iter()
            .filter(|(_, _, _, e)| *e <= max_deviation)
            .collect();
        if !valid_max.is_empty() {
            valid_max.iter().min_by_key(|(n, _, _, _)| n).unwrap()
        } else {
            all_results.last().unwrap()
        }
    };

    if verbose {
        println!(
            "  Selected: {} colors (max_err={:.1}, dev_rate={:.1}%)",
            best.0,
            best.3,
            best.2 * 100.0
        );
    }

    best.1.iter().map(|c| [c[0], c[1], c[2], 255]).collect()
}

fn main() {
    let args = Args::parse();

    let mut img = image::open(&args.input)
        .expect("Failed to open input image")
        .to_rgba8();

    let (width, height) = img.dimensions();
    if !args.quiet {
        println!("Input image: {}x{}", width, height);
    }

    // Auto-scale to get cell size into optimal range (8-16 pixels)
    let min_cell_size = 8.0;
    let max_cell_size = 16.0;
    let mut scale_factor = 1u32;

    let mut grid = find_best_grid(&img);
    let mut avg_cell = (grid.cell_w + grid.cell_h) / 2.0;

    while avg_cell < min_cell_size && scale_factor < 16 {
        scale_factor *= 2;
        let new_width = width * scale_factor;
        let new_height = height * scale_factor;
        img = image::imageops::resize(&img, new_width, new_height, image::imageops::FilterType::Nearest);
        grid = find_best_grid(&img);
        avg_cell = (grid.cell_w + grid.cell_h) / 2.0;
    }

    while avg_cell > max_cell_size && scale_factor > 1 {
        scale_factor /= 2;
        let new_width = width * scale_factor;
        let new_height = height * scale_factor;
        img = image::imageops::resize(&img, new_width, new_height, image::imageops::FilterType::Lanczos3);
        grid = find_best_grid(&img);
        avg_cell = (grid.cell_w + grid.cell_h) / 2.0;
    }

    if scale_factor != 1 && !args.quiet {
        println!("Auto-scaled by {}x to get cell size ~{:.1}px", scale_factor, avg_cell);
    }

    let (width, height) = img.dimensions();

    if !args.quiet {
        println!(
            "Best grid: {}x{} (cell size ~{:.1}x{:.1}px, offset {:.1},{:.1})",
            grid.n_cols, grid.n_rows, grid.cell_w, grid.cell_h, grid.offset_x, grid.offset_y
        );
        println!("Output size: {}x{}", grid.n_cols, grid.n_rows);
    }

    // Apply median filter for denoising (match Python)
    let denoise_size = {
        let min_cell = grid.cell_w.min(grid.cell_h);
        let size = (min_cell / 6.0) as usize;
        let size = size.max(1);
        // Make odd
        if size % 2 == 0 { size + 1 } else { size }
    };
    let denoised = apply_median_filter(&img, denoise_size);
    if !args.quiet && denoise_size > 1 {
        println!("Applied median filter (size={}) for color sampling", denoise_size);
    }

    // Sample colors at grid centers
    let mut sampled: Vec<(u32, u32, [u8; 4])> = Vec::new();

    for row in 0..grid.n_rows {
        for col in 0..grid.n_cols {
            let cx = if grid.start_partial_x && col == 0 {
                (grid.offset_x / 2.0) as u32
            } else {
                let adj_col = col - if grid.start_partial_x { 1 } else { 0 };
                (grid.offset_x + (adj_col as f64 + 0.5) * grid.cell_w) as u32
            };

            let cy = if grid.start_partial_y && row == 0 {
                (grid.offset_y / 2.0) as u32
            } else {
                let adj_row = row - if grid.start_partial_y { 1 } else { 0 };
                (grid.offset_y + (adj_row as f64 + 0.5) * grid.cell_h) as u32
            };

            let cx = cx.clamp(0, width - 1);
            let cy = cy.clamp(0, height - 1);

            let color = sample_color(&denoised, cx, cy, grid.cell_w, grid.cell_h);
            sampled.push((col as u32, row as u32, color));
        }
    }

    // Store original sampled colors for max_deviation checks
    let original_colors: HashMap<(u32, u32), [u8; 4]> = sampled
        .iter()
        .map(|&(col, row, color)| ((col, row), color))
        .collect();

    // Max deviation thresholds
    let max_deviation = 12.0; // Strict limit for common colors
    let max_deviation_rare = 22.0; // Relaxed limit for rare colors

    // Helper to check if color is within allowed deviation
    let is_valid_color = |pos: (u32, u32), new_color: [u8; 4], is_rare: bool| -> bool {
        let orig = original_colors[&pos];
        let effective_max = if is_rare { max_deviation_rare } else { max_deviation };
        color_distance([orig[0], orig[1], orig[2]], [new_color[0], new_color[1], new_color[2]]) <= effective_max
    };

    // Spatial clustering
    if !args.quiet {
        let unique_before: HashSet<_> = sampled.iter().map(|s| s.2).collect();
        println!(
            "Spatial clustering ({} unique colors)...",
            unique_before.len()
        );
    }
    precluster_colors_spatial(&mut sampled, 8.0, 12.0);
    if !args.quiet {
        let unique_after: HashSet<_> = sampled.iter().map(|s| s.2).collect();
        println!(
            "  {} unique colors after spatial clustering",
            unique_after.len()
        );
    }

    // Validate spatial clustering didn't exceed max_deviation (strict check)
    let mut spatial_reverts = 0;
    for (col, row, color) in sampled.iter_mut() {
        if !is_valid_color((*col, *row), *color, false) {
            *color = original_colors[&(*col, *row)];
            spatial_reverts += 1;
        }
    }
    if spatial_reverts > 0 && !args.quiet {
        println!("  Reverted {} pixels that exceeded max_deviation", spatial_reverts);
    }

    // Extract palette with quality metrics
    let colors: Vec<[u8; 4]> = sampled.iter().map(|s| s.2).collect();
    let palette = extract_palette_quality(&colors, !args.quiet);
    if !args.quiet {
        println!("Palette: {} colors", palette.len());
    }

    // Snap all colors to palette
    let mut snapped_grid: HashMap<(u32, u32), [u8; 4]> = HashMap::new();
    for &(col, row, color) in &sampled {
        let snapped = snap_to_palette(color, &palette);
        snapped_grid.insert((col, row), snapped);
    }

    // Compute rarity threshold based on median palette usage / 4
    let mut color_counts: HashMap<[u8; 4], usize> = HashMap::new();
    for &color in snapped_grid.values() {
        *color_counts.entry(color).or_insert(0) += 1;
    }

    let mut usage_values: Vec<usize> = color_counts.values().copied().collect();
    usage_values.sort();
    let median_usage = if usage_values.is_empty() { 10 } else { usage_values[usage_values.len() / 2] };
    let rarity_threshold = 3.max(median_usage / 4);

    if !args.quiet {
        println!("  Rarity threshold: {} (median usage: {})", rarity_threshold, median_usage);
    }

    // Rare color cleanup with relaxed threshold for rare colors
    let rare_color_merge_threshold = 25.0;

    let rare_colors: HashSet<_> = color_counts
        .iter()
        .filter(|&(_, count)| *count < rarity_threshold)
        .map(|(c, _)| *c)
        .collect();

    if !rare_colors.is_empty() && !args.quiet {
        println!(
            "  Found {} rare colors (used <{} times)",
            rare_colors.len(),
            rarity_threshold
        );
    }

    let mut rare_mapping: HashMap<[u8; 4], [u8; 4]> = HashMap::new();
    for &rare in &rare_colors {
        let mut best_dist = f64::INFINITY;
        let mut best_color = rare;
        for &other in &palette {
            if !rare_colors.contains(&other) && other != rare {
                let dist = color_distance([rare[0], rare[1], rare[2]], [other[0], other[1], other[2]]);
                if dist < best_dist {
                    best_dist = dist;
                    best_color = other;
                }
            }
        }
        if best_dist <= rare_color_merge_threshold {
            rare_mapping.insert(rare, best_color);
        }
    }

    if !rare_mapping.is_empty() && !args.quiet {
        println!(
            "  Merging {} rare colors into nearby palette colors",
            rare_mapping.len()
        );
    }

    // Apply rare color mapping with relaxed max_deviation check (rare colors get more tolerance)
    let mut rare_blocked = 0;
    let positions: Vec<(u32, u32)> = snapped_grid.keys().copied().collect();
    for pos in positions {
        let color = snapped_grid[&pos];
        if let Some(&new_color) = rare_mapping.get(&color) {
            // Rare colors get relaxed threshold
            if is_valid_color(pos, new_color, true) {
                snapped_grid.insert(pos, new_color);
            } else {
                rare_blocked += 1;
            }
        }
    }
    if rare_blocked > 0 && !args.quiet {
        println!("    Blocked {} merges (exceeded even relaxed threshold)", rare_blocked);
    }

    // Local smoothing
    let similar_neighbor_threshold = 12.0;
    let mut smoothed = 0;
    let positions: Vec<(u32, u32)> = snapped_grid.keys().copied().collect();

    for (col, row) in positions {
        let color = snapped_grid[&(col, row)];

        // Get 3x3 neighborhood
        let mut neighbors: Vec<[u8; 4]> = Vec::new();
        for dc in -1i32..=1 {
            for dr in -1i32..=1 {
                let nc = col as i32 + dc;
                let nr = row as i32 + dr;
                if nc >= 0 && nr >= 0 {
                    if let Some(&c) = snapped_grid.get(&(nc as u32, nr as u32)) {
                        neighbors.push(c);
                    }
                }
            }
        }

        // Check if any neighbor is super close
        let has_similar = neighbors.iter().any(|&n| {
            n != color
                && color_distance([color[0], color[1], color[2]], [n[0], n[1], n[2]])
                    < similar_neighbor_threshold
        });

        if has_similar {
            let mut local_counts: HashMap<[u8; 4], usize> = HashMap::new();
            for &n in &neighbors {
                *local_counts.entry(n).or_insert(0) += 1;
            }

            let mut best_color = color;
            let mut best_count = *local_counts.get(&color).unwrap_or(&0);

            for (&c, &count) in &local_counts {
                if count > best_count
                    && color_distance([color[0], color[1], color[2]], [c[0], c[1], c[2]])
                        < similar_neighbor_threshold
                {
                    best_color = c;
                    best_count = count;
                }
            }

            if best_color != color {
                // Strict check for local smoothing
                if is_valid_color((col, row), best_color, false) {
                    snapped_grid.insert((col, row), best_color);
                    smoothed += 1;
                }
            }
        }
    }

    if smoothed > 0 && !args.quiet {
        println!("  Smoothed {} pixels to match local neighbors", smoothed);
    }

    // Adjacency-aware artifact merge
    let adjacency_merge_threshold = 18.0;
    let used_colors: HashSet<_> = snapped_grid.values().copied().collect();
    let palette_list: Vec<_> = used_colors.into_iter().collect();

    let mut palette_usage: HashMap<[u8; 4], usize> = HashMap::new();
    for &c in snapped_grid.values() {
        *palette_usage.entry(c).or_insert(0) += 1;
    }

    // Build adjacency and isolation info
    let mut color_neighbors: HashMap<[u8; 4], HashSet<[u8; 4]>> = HashMap::new();
    let mut isolated_pixels: HashMap<[u8; 4], usize> = HashMap::new();
    let mut total_pixels: HashMap<[u8; 4], usize> = HashMap::new();

    for &c in &palette_list {
        color_neighbors.insert(c, HashSet::new());
        isolated_pixels.insert(c, 0);
        total_pixels.insert(c, 0);
    }

    for (&(col, row), &color) in &snapped_grid {
        *total_pixels.entry(color).or_insert(0) += 1;
        let mut has_same = false;

        for (dc, dr) in [(-1i32, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
            let nc = col as i32 + dc;
            let nr = row as i32 + dr;
            if nc >= 0 && nr >= 0 {
                if let Some(&neighbor_color) = snapped_grid.get(&(nc as u32, nr as u32)) {
                    if neighbor_color == color {
                        has_same = true;
                    } else if dc == 0 || dr == 0 {
                        color_neighbors
                            .entry(color)
                            .or_default()
                            .insert(neighbor_color);
                    }
                }
            }
        }

        if !has_same {
            *isolated_pixels.entry(color).or_insert(0) += 1;
        }
    }

    let mut adjacency_merges: HashMap<[u8; 4], [u8; 4]> = HashMap::new();

    for &color in &palette_list {
        if adjacency_merges.contains_key(&color) {
            continue;
        }
        if let Some(neighbors) = color_neighbors.get(&color) {
            for &neighbor in neighbors {
                if adjacency_merges.contains_key(&neighbor) {
                    continue;
                }
                let dist =
                    color_distance([color[0], color[1], color[2]], [neighbor[0], neighbor[1], neighbor[2]]);
                if dist < adjacency_merge_threshold {
                    let usage_color = *palette_usage.get(&color).unwrap_or(&0);
                    let usage_neighbor = *palette_usage.get(&neighbor).unwrap_or(&0);
                    let (rare_color, common_color) = if usage_color < usage_neighbor {
                        (color, neighbor)
                    } else {
                        (neighbor, color)
                    };

                    let total = *total_pixels.get(&rare_color).unwrap_or(&0);
                    let isolated = *isolated_pixels.get(&rare_color).unwrap_or(&0);

                    if total > 0 {
                        let isolation_rate = isolated as f64 / total as f64;
                        if isolation_rate >= 0.7 {
                            if let Some(rare_neighbors) = color_neighbors.get(&rare_color) {
                                if rare_neighbors.contains(&common_color) {
                                    adjacency_merges.insert(rare_color, common_color);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if !adjacency_merges.is_empty() && !args.quiet {
        println!(
            "  Merged {} artifact colors (isolated/always adjacent to similar)",
            adjacency_merges.len()
        );
    }

    // Apply adjacency merges with rarity-based tolerance
    let mut adj_blocked = 0;
    let positions: Vec<(u32, u32)> = snapped_grid.keys().copied().collect();
    for pos in positions {
        let color = snapped_grid[&pos];
        if let Some(&new_color) = adjacency_merges.get(&color) {
            let pixel_count = *total_pixels.get(&color).unwrap_or(&0);
            let is_rare = pixel_count < rarity_threshold;
            if is_valid_color(pos, new_color, is_rare) {
                snapped_grid.insert(pos, new_color);
            } else {
                adj_blocked += 1;
            }
        }
    }
    if adj_blocked > 0 && !args.quiet {
        println!("    Blocked {} merges that would exceed max_deviation", adj_blocked);
    }

    // Global palette merge
    let palette_merge_threshold = 15.0;
    let used_colors: HashSet<_> = snapped_grid.values().copied().collect();
    let mut palette_list: Vec<_> = used_colors.into_iter().collect();

    let mut palette_usage: HashMap<[u8; 4], usize> = HashMap::new();
    for &c in snapped_grid.values() {
        *palette_usage.entry(c).or_insert(0) += 1;
    }

    palette_list.sort_by(|a, b| {
        palette_usage
            .get(b)
            .unwrap_or(&0)
            .cmp(palette_usage.get(a).unwrap_or(&0))
    });

    let mut final_palette: Vec<[u8; 4]> = Vec::new();
    let mut merged_map: HashMap<[u8; 4], [u8; 4]> = HashMap::new();

    for &color in &palette_list {
        let mut merged = false;
        for &existing in &final_palette {
            if color_distance(
                [color[0], color[1], color[2]],
                [existing[0], existing[1], existing[2]],
            ) < palette_merge_threshold
            {
                merged_map.insert(color, existing);
                merged = true;
                break;
            }
        }
        if !merged {
            final_palette.push(color);
            merged_map.insert(color, color);
        }
    }

    if final_palette.len() < palette_list.len() && !args.quiet {
        println!(
            "  Merged {} similar palette colors",
            palette_list.len() - final_palette.len()
        );
    }

    // Apply global palette merge with rarity-based tolerance
    let mut merge_blocked = 0;
    let positions: Vec<(u32, u32)> = snapped_grid.keys().copied().collect();
    for pos in positions {
        let color = snapped_grid[&pos];
        if let Some(&new_color) = merged_map.get(&color) {
            if new_color != color {
                let is_rare = *palette_usage.get(&color).unwrap_or(&0) < rarity_threshold;
                if is_valid_color(pos, new_color, is_rare) {
                    snapped_grid.insert(pos, new_color);
                } else {
                    merge_blocked += 1;
                }
            }
        }
    }
    if merge_blocked > 0 && !args.quiet {
        println!("    Blocked {} palette merges that would exceed max_deviation", merge_blocked);
    }

    // Palette optimization: adjust each palette color to be the centroid
    // of the original colors of all pixels assigned to it
    if !args.quiet {
        println!("  Optimizing palette colors to minimize error...");
    }

    // Get current palette
    let used_colors: HashSet<_> = snapped_grid.values().copied().collect();
    let current_palette: Vec<_> = used_colors.into_iter().collect();

    // Group positions by their current palette color
    let mut color_to_positions: HashMap<[u8; 4], Vec<(u32, u32)>> = HashMap::new();
    for (&pos, &color) in &snapped_grid {
        color_to_positions.entry(color).or_default().push(pos);
    }

    // Compute optimal color for each palette entry
    let mut color_remap: HashMap<[u8; 4], [u8; 4]> = HashMap::new();
    for &palette_color in &current_palette {
        let positions = color_to_positions.get(&palette_color).cloned().unwrap_or_default();
        if positions.is_empty() {
            color_remap.insert(palette_color, palette_color);
            continue;
        }

        // Compute average of original colors
        let mut sum_r = 0i32;
        let mut sum_g = 0i32;
        let mut sum_b = 0i32;
        for &pos in &positions {
            let orig = original_colors[&pos];
            sum_r += orig[0] as i32;
            sum_g += orig[1] as i32;
            sum_b += orig[2] as i32;
        }
        let n = positions.len() as i32;
        let optimal_color = [(sum_r / n) as u8, (sum_g / n) as u8, (sum_b / n) as u8, 255];

        // Verify this doesn't exceed max_deviation for any pixel
        let all_valid = positions.iter().all(|&pos| {
            let orig = original_colors[&pos];
            color_distance([orig[0], orig[1], orig[2]], [optimal_color[0], optimal_color[1], optimal_color[2]]) <= max_deviation
        });

        if all_valid {
            color_remap.insert(palette_color, optimal_color);
        } else {
            color_remap.insert(palette_color, palette_color);
        }
    }

    // Apply optimized colors
    for (_, color) in snapped_grid.iter_mut() {
        if let Some(&new_color) = color_remap.get(color) {
            *color = new_color;
        }
    }

    // Final pass: re-snap each pixel to the closest palette color based on its
    // original sampled color
    let final_palette: Vec<[u8; 4]> = snapped_grid.values().copied().collect::<HashSet<_>>().into_iter().collect();
    let mut resnapped = 0;

    let positions: Vec<(u32, u32)> = snapped_grid.keys().copied().collect();
    for pos in positions {
        let orig_color = original_colors[&pos];
        let current = snapped_grid[&pos];

        let mut best_color = current;
        let mut best_dist = color_distance([orig_color[0], orig_color[1], orig_color[2]], [current[0], current[1], current[2]]);

        for &p in &final_palette {
            let dist = color_distance([orig_color[0], orig_color[1], orig_color[2]], [p[0], p[1], p[2]]);
            if dist < best_dist {
                best_dist = dist;
                best_color = p;
            }
        }

        if best_color != current {
            snapped_grid.insert(pos, best_color);
            resnapped += 1;
        }
    }

    if resnapped > 0 && !args.quiet {
        println!("  Re-snapped {} pixels to closest palette color", resnapped);
    }

    // Update final palette
    let final_colors: HashSet<_> = snapped_grid.values().copied().collect();
    if !args.quiet {
        println!("  Final palette: {} colors", final_colors.len());
    }

    // Create output
    let mut output: RgbaImage = ImageBuffer::new(grid.n_cols as u32, grid.n_rows as u32);
    for (&(col, row), &color) in &snapped_grid {
        output.put_pixel(col, row, Rgba(color));
    }

    let output_path = args.output.unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap().to_str().unwrap();
        args.input.with_file_name(format!("{}_pixels.png", stem))
    });

    output.save(&output_path).expect("Failed to save output");
    if !args.quiet {
        println!("Saved to: {}", output_path.display());
    }
}
