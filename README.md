# restore-pixel-art

Restore clean pixel art from degraded sources:

- **AI-generated pixel art** - Extract clean pixels from upscaled/anti-aliased AI outputs
- **JPEG-compressed pixel art** - Recover sharp pixels from lossy compression artifacts
- **Upscaled pixel art** - Detect the original pixel grid and sample true colors

## Features

- Automatic grid detection using autocorrelation and edge signal analysis
- Spatial color clustering to reduce AI noise
- Smart palette extraction with quality metrics
- Automatic cleanup of rare/artifact colors
- Adjacency-aware merging to remove compression artifacts

## Demo

Restoring pixel art from JPG:

<img width="855" height="1002" alt="ezgif com-apng-maker" src="https://github.com/user-attachments/assets/c16b7dcc-2f49-4d8f-8b2b-b8d8b71324fe" />
<img width="840" height="990" alt="ezgif com-apng-maker (3)" src="https://github.com/user-attachments/assets/821364f0-07c8-4067-ac1a-6bc90a9a6b7c" />

Nano Banana 2 image to pixel art:

<img width="864" height="672" alt="ezgif com-apng-maker (1)" src="https://github.com/user-attachments/assets/e51f1513-80d4-4247-8c9b-4f3bd9a89055" />

Older AI generated pixel art:

<img width="799" height="592" alt="ezgif com-apng-maker (2)" src="https://github.com/user-attachments/assets/348b8dfc-1c5d-4407-b440-5f2c893bb5bb" />


## Installation

```bash
uv sync
```

## Usage

```bash
# Basic usage
uv run restore-pixel-art input.png -o output.png

# Stop after grid detection (debug centers)
uv run restore-pixel-art input.png -o output.png --pass0-only

# Stop after color sampling (before palette reduction)
uv run restore-pixel-art input.png -o output.png --pass1-only

# Full debug output
uv run restore-pixel-art input.png -o output.png --debug
```

## Output Files

- `output.png` - Final restored pixel art with optimized palette
- `output_pass0.png` - Original image with detected pixel centers marked
- `output_pass1.png` - True sampled colors before palette reduction
- `output_initial_grid.png` - Debug overlay showing detected grid (with --debug)

## Requirements

- Python 3.10+
- Pillow
- NumPy
- SciPy

## See also

* [Unfaker](https://jenissimo.itch.io/unfaker)
