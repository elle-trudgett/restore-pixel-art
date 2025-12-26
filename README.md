# restore-pixel-art

Restore clean pixel art from degraded sources:

- **AI-generated pixel art** - Extract clean pixels from upscaled/anti-aliased AI outputs
- **JPEG-compressed pixel art** - Recover sharp pixels from lossy compression artifacts
- **Upscaled pixel art** - Detect the original pixel grid and sample true colors

## Features

- Automatic grid detection using autocorrelation
- Edge-based pixel center refinement for sub-pixel accuracy
- Confidence-based iterative grid placement
- Spatial color clustering to reduce AI noise
- Smart palette extraction with quality metrics
- Automatic cleanup of rare/similar colors

## Usage

```bash
# Basic usage
uv run extract.py input.png -o output.png

# Stop after grid detection (debug centers)
uv run extract.py input.png -o output.png --pass0-only

# Stop after color sampling (before palette reduction)
uv run extract.py input.png -o output.png --pass1-only

# Full debug output
uv run extract.py input.png -o output.png --debug
```

## Output Files

- `output.png` - Final restored pixel art with optimized palette
- `output_pass0.png` - Original image with detected pixel centers marked
- `output_pass1.png` - True sampled colors before palette reduction

## Requirements

- Python 3.10+
- Pillow
- NumPy
- scikit-learn

## Installation

```bash
uv sync
```
