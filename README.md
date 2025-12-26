# restore-pixel-art

Restore clean pixel art from degraded sources:

- **AI-generated pixel art** - Extract clean pixels from upscaled/anti-aliased AI outputs
- **JPEG-compressed pixel art** - Recover sharp pixels from lossy compression artifacts
- **Upscaled pixel art** - Detect the original pixel grid and sample true colors

## Features

- Automatic grid detection using autocorrelation - pixels don't need to fall on a strict grid
- Edge-based pixel center refinement for sub-pixel accuracy
- Confidence-based iterative grid placement
- Spatial color clustering to reduce AI noise
- Smart palette extraction with quality metrics
- Automatic cleanup of rare/similar colors

## Example

**Input image size**: 528 x 297 .jpg

**Output image size**: 100 x 56 .png

Note! Higher resolution images work better. This was chosen as a stress test for the algorithm.

| Before | After |
|--------|-------|
| <img height="600px" alt="Blurry pixel art jpg with wonky pixels" src="https://github.com/user-attachments/assets/e70e4118-f7fd-4368-9471-e3cff9592499" /> | <img height="600px" alt="Sharp pixel art character" src="https://github.com/user-attachments/assets/a39ac524-f4c4-425a-b86e-43248fe5b954" /> |
| <img height="600px" alt="Blurry pixels with noise" src="https://github.com/user-attachments/assets/a0ebf3b9-dede-43f0-b877-3a2f330ee4e4" /> | <img height="600px" alt="Sharp pixels" src="https://github.com/user-attachments/assets/52a56c39-1a3f-490a-97c9-d7cf8d440600" /> |

### Steps

| Step | Result |
|------|--------|
| **Example blurry pixel art image (.jpg)** (Scaled down for illustration) | <img height="320" alt="Blurry pixel art image jpg" src="https://github.com/user-attachments/assets/9d93978d-59a6-4346-acb1-cb417ada1a33" /> |
| **Zoomed in to show blurry details** | <img height="320" alt="Blurry edges of pixels can be seen" src="https://github.com/user-attachments/assets/00a83bbe-0d92-459f-9dc1-983caa7ecce8" /> |
| **Iterative confidence-based pixel centroid and grid detection** | <img height="320" alt="Edge detection on high confidence pixels" src="https://github.com/user-attachments/assets/4ef39db7-da61-4be5-9b51-10f80545de64" /> |
| **Final pixel grid determined** | <img height="320" alt="Showing centers of all pixels" src="https://github.com/user-attachments/assets/b68bbde5-201f-4feb-b4ef-66e6c76469e9" /> |
| **Color sampling to convert to pure pixels** | <img height="320" alt="Convert to 1:1 pixel size" src="https://github.com/user-attachments/assets/bdd87c63-0d51-4425-b561-bdc590ef48bc" /> |
| **Palette optimization (from 1244 → 43 colors)** | <img height="320" alt="Palette reduced without sacrificing quality" src="https://github.com/user-attachments/assets/407d58e6-9620-42b7-b62e-bac19e74e8af" /> |
| **Final result** | <img height="320" alt="Pixel art character" src="https://github.com/user-attachments/assets/ef946cb9-ee6f-48d0-a72f-d070d59971cd" /> |


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

## See also

* [Unfaker](https://jenissimo.itch.io/unfaker)
