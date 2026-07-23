#!/usr/bin/env python3
"""
Extract Lead I from fixed-layout, single-page ECG PDF reports.

Each PDF is:

1. Checked to ensure that it contains exactly one page.
2. Rendered as a PNG image at a specified DPI.
3. Optionally checked for color content.
4. Cropped using fixed pixel coordinates.
5. Saved as a Lead I PNG image.

The crop coordinates are specific to the ECG report layout and rendering DPI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Lead I from fixed-layout ECG PDF reports."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="CSV containing the ECG PDF paths.",
    )

    parser.add_argument(
        "--path-column",
        required=True,
        help="Name of the CSV column containing PDF paths.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for cropped Lead I PNG images.",
    )

    parser.add_argument(
        "--left",
        required=True,
        type=int,
        help="Left crop coordinate in pixels.",
    )

    parser.add_argument(
        "--top",
        required=True,
        type=int,
        help="Top crop coordinate in pixels.",
    )

    parser.add_argument(
        "--right",
        required=True,
        type=int,
        help="Right crop coordinate in pixels.",
    )

    parser.add_argument(
        "--bottom",
        required=True,
        type=int,
        help="Bottom crop coordinate in pixels.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF rendering resolution.",
    )

    parser.add_argument(
        "--require-color",
        action="store_true",
        help="Skip reports that contain no colored pixels.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )

    return parser.parse_args()


def is_single_page_pdf(pdf_path: Path) -> bool:
    """Return True when the PDF contains exactly one page."""
    try:
        with pdf_path.open("rb") as file:
            reader = PdfReader(file)
            return len(reader.pages) == 1
    except Exception as error:
        print(f"Could not read {pdf_path}: {error}")
        return False


def has_color_content(image: Image.Image) -> bool:
    """
    Determine whether an image contains pixels with unequal RGB channels.

    A grayscale image represented in RGB has identical red, green, and blue
    channel values for every pixel.
    """
    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image)

    red = array[..., 0]
    green = array[..., 1]
    blue = array[..., 2]

    return bool(
        np.any(
            (red != green)
            | (green != blue)
            | (red != blue)
        )
    )


def validate_crop_coordinates(
    image: Image.Image,
    crop_coordinates: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = crop_coordinates
    width, height = image.size

    if left < 0 or top < 0:
        raise ValueError("Crop coordinates cannot be negative.")

    if right <= left:
        raise ValueError("The right coordinate must exceed the left coordinate.")

    if bottom <= top:
        raise ValueError(
            "The bottom coordinate must exceed the top coordinate."
        )

    if right > width or bottom > height:
        raise ValueError(
            "Crop coordinates exceed the rendered image dimensions: "
            f"image={width}x{height}, crop={crop_coordinates}"
        )


def process_pdf(
    pdf_path: Path,
    output_path: Path,
    crop_coordinates: tuple[int, int, int, int],
    dpi: int,
    require_color: bool,
) -> str:
    """
    Render and crop one ECG report.

    Returns one of:
    - processed
    - multipage
    - grayscale
    - failed
    """
    if not is_single_page_pdf(pdf_path):
        print(f"Skipping non-single-page PDF: {pdf_path}")
        return "multipage"

    try:
        pages = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=1,
            last_page=1,
        )

        if not pages:
            print(f"No rendered page was returned: {pdf_path}")
            return "failed"

        image = pages[0].convert("RGB")

        if require_color and not has_color_content(image):
            print(f"Skipping grayscale PDF: {pdf_path}")
            return "grayscale"

        validate_crop_coordinates(
            image=image,
            crop_coordinates=crop_coordinates,
        )

        lead1_image = image.crop(crop_coordinates)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        lead1_image.save(
            output_path,
            format="PNG",
        )

        print(f"Saved: {output_path}")
        return "processed"

    except Exception as error:
        print(f"Failed to process {pdf_path}: {error}")
        return "failed"


def main() -> None:
    args = parse_arguments()

    if not args.input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist: {args.input_csv}"
        )

    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")

    crop_coordinates = (
        args.left,
        args.top,
        args.right,
        args.bottom,
    )

    dataframe = pd.read_csv(args.input_csv)

    if args.path_column not in dataframe.columns:
        raise KeyError(
            f"Column '{args.path_column}' was not found in "
            f"{args.input_csv}."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = {
        "processed": 0,
        "multipage": 0,
        "grayscale": 0,
        "failed": 0,
        "missing": 0,
        "existing": 0,
    }

    paths = dataframe[args.path_column].dropna()

    for raw_path in paths:
        pdf_path = Path(str(raw_path))

        if not pdf_path.exists():
            counts["missing"] += 1
            print(f"File not found: {pdf_path}")
            continue

        output_path = args.output_dir / f"{pdf_path.stem}.png"

        if output_path.exists() and not args.overwrite:
            counts["existing"] += 1
            continue

        status = process_pdf(
            pdf_path=pdf_path,
            output_path=output_path,
            crop_coordinates=crop_coordinates,
            dpi=args.dpi,
            require_color=args.require_color,
        )

        counts[status] += 1

    print()
    print("Lead I extraction summary")
    print("-------------------------")
    print(f"Successfully processed: {counts['processed']:,}")
    print(f"Existing outputs skipped: {counts['existing']:,}")
    print(f"Multipage PDFs skipped: {counts['multipage']:,}")
    print(f"Grayscale PDFs skipped: {counts['grayscale']:,}")
    print(f"Missing files: {counts['missing']:,}")
    print(f"Failed: {counts['failed']:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
