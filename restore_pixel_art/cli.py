"""Command-line interface for restore-pixel-art."""

import argparse
from pathlib import Path

from .core import extract_pixel_art


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
