#!/usr/bin/env python3
"""
Generate standardized ECG images from MIMIC-IV-ECG WFDB records.

The script:

1. Reads an ECG–NT-proBNP matched cohort CSV.
2. Locates each ECG waveform using its study_id.
3. Reads the WFDB waveform.
4. Applies baseline correction using a high-pass Butterworth filter.
5. Generates a standardized 12-lead ECG image.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt


DEFAULT_SAMPLING_RATE = 500.0
DEFAULT_MM_PER_MV = 10.0
DEFAULT_MM_PER_SECOND = 25.0
DEFAULT_BASELINE_CUTOFF_HZ = 0.67


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ECG images from matched MIMIC-IV WFDB records."
    )

    parser.add_argument(
        "--matched-csv",
        required=True,
        type=Path,
        help="CSV produced by 01_match_ecg_ntprobnp.py.",
    )

    parser.add_argument(
        "--ecg-root",
        required=True,
        type=Path,
        help="Root directory containing the MIMIC-IV-ECG files folder.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which generated ECG images will be saved.",
    )

    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=DEFAULT_SAMPLING_RATE,
        help=(
            "Expected ECG sampling frequency. The WFDB record sampling "
            "frequency is used when available."
        ),
    )

    parser.add_argument(
        "--baseline-cutoff",
        type=float,
        default=DEFAULT_BASELINE_CUTOFF_HZ,
        help="High-pass cutoff frequency in Hz.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Output image resolution.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images that already exist.",
    )

    return parser.parse_args()


def normalize_lead_name(name: str) -> str:
    """Normalize common WFDB ECG lead-name variations."""
    cleaned = name.strip()

    replacements = {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
    }

    return replacements.get(cleaned.upper(), cleaned)


def read_wfdb_record(
    record_path: Path,
) -> tuple[pd.DataFrame, float]:
    """Read a WFDB record and return its waveform dataframe and frequency."""
    record = wfdb.rdrecord(str(record_path))

    lead_names = [normalize_lead_name(name) for name in record.sig_name]

    ecg = pd.DataFrame(
        record.p_signal,
        columns=lead_names,
    )

    ecg = ecg.apply(pd.to_numeric, errors="coerce")
    ecg = ecg.dropna(axis=1, how="all")
    ecg = ecg.dropna(axis=0, how="all")

    sampling_rate = float(record.fs)

    return ecg, sampling_rate


def apply_baseline_correction(
    ecg: pd.DataFrame,
    sampling_rate: float,
    cutoff_hz: float,
) -> pd.DataFrame:
    """Apply a second-order high-pass Butterworth filter to each ECG lead."""
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    nyquist = 0.5 * sampling_rate

    if not 0 < cutoff_hz < nyquist:
        raise ValueError(
            f"Baseline cutoff must be between 0 and {nyquist:g} Hz."
        )

    normalized_cutoff = cutoff_hz / nyquist

    b, a = butter(
        N=2,
        Wn=normalized_cutoff,
        btype="highpass",
        analog=False,
    )

    corrected = ecg.copy()

    for column in corrected.columns:
        signal = corrected[column].astype(float)

        # Interpolate isolated missing samples before filtering.
        signal = signal.interpolate(
            method="linear",
            limit_direction="both",
        )

        if signal.isna().any():
            continue

        minimum_length = 3 * max(len(a), len(b))

        if len(signal) <= minimum_length:
            continue

        corrected[column] = filtfilt(
            b,
            a,
            signal.to_numpy(),
        )

    return corrected


def locate_record(ecg_root: Path, subject_id: int, study_id: int) -> Path | None:
    """
    Locate the WFDB header corresponding to a subject and study.

    A subject-specific search is attempted first, followed by a general
    study-specific fallback.
    """
    subject_text = str(subject_id)
    study_text = str(study_id)

    subject_prefix = f"p{subject_text[:4]}"
    subject_folder = f"p{subject_text}"
    study_folder = f"s{study_text}"

    direct_study_dir = (
        ecg_root
        / subject_prefix
        / subject_folder
        / study_folder
    )

    if direct_study_dir.exists():
        headers = sorted(direct_study_dir.glob("*.hea"))
        if headers:
            return headers[0].with_suffix("")

    patterns = [
        str(
            ecg_root
            / "p*"
            / subject_folder
            / study_folder
            / "*.hea"
        ),
        str(
            ecg_root
            / "p*"
            / "*"
            / study_folder
            / "*.hea"
        ),
    ]

    for pattern in patterns:
        matches = sorted(glob.glob(pattern))

        if matches:
            return Path(matches[0]).with_suffix("")

    return None


def configure_grid(
    axis: plt.Axes,
    total_samples: int,
    sampling_rate: float,
) -> None:
    axis.set_xlim(0, total_samples)

    samples_per_mm = sampling_rate / DEFAULT_MM_PER_SECOND
    small_x_step = samples_per_mm
    large_x_step = small_x_step * 5

    small_y_step = 0.1
    large_y_step = 0.5

    axis.set_xticks(
        np.arange(0, total_samples + 1, large_x_step)
    )

    axis.set_xticks(
        np.arange(0, total_samples + 1, small_x_step),
        minor=True,
    )

    y_min, y_max = axis.get_ylim()

    axis.set_yticks(
        np.arange(y_min, y_max + large_y_step, large_y_step)
    )

    axis.set_yticks(
        np.arange(y_min, y_max + small_y_step, small_y_step),
        minor=True,
    )

    axis.grid(
        which="major",
        color="red",
        linestyle="-",
        linewidth=0.4,
    )

    axis.grid(
        which="minor",
        color="red",
        linestyle=":",
        linewidth=0.2,
    )

    axis.tick_params(
        which="both",
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )


def add_lead_label(
    axis: plt.Axes,
    x_position: float,
    y_position: float,
    lead_name: str,
) -> None:
    axis.text(
        x_position,
        y_position,
        lead_name,
        ha="left",
        va="top",
        weight="bold",
        fontsize=9.5,
    )


def plot_ecg(
    ecg: pd.DataFrame,
    sampling_rate: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Create a standardized 12-lead ECG image."""
    required_layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    rhythm_strips = ["V1", "II", "V5"]

    available_leads = {
        lead
        for row in required_layout
        for lead in row
        if lead in ecg.columns
    }

    if not available_leads:
        raise ValueError("No expected ECG leads were found in the record.")

    total_samples = len(ecg)

    if total_samples == 0:
        raise ValueError("The ECG record contains no valid samples.")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 9.5

    figure = plt.figure(
        figsize=(11, 8.5),
        dpi=dpi,
    )

    left = 0.17
    bottom = 8.5 - 7.85
    width = 11 - (2 * left)
    height = 7.85 - 2.3

    axis = figure.add_axes(
        [
            left / 11,
            bottom / 8.5,
            width / 11,
            height / 8.5,
        ]
    )

    y_max = height * 25.4 / DEFAULT_MM_PER_MV
    axis.set_ylim(0, y_max)

    configure_grid(
        axis=axis,
        total_samples=total_samples,
        sampling_rate=sampling_rate,
    )

    number_of_columns = 4
    segment_length = total_samples // number_of_columns

    total_rows = 6
    y_offset = axis.get_ylim()[1] / total_rows

    # Standard 12-lead display.
    for row_index, lead_row in enumerate(required_layout):
        for column_index, lead_name in enumerate(lead_row):
            if lead_name not in ecg.columns:
                continue

            segment_start = column_index * segment_length

            if column_index == number_of_columns - 1:
                segment_end = total_samples
            else:
                segment_end = (column_index + 1) * segment_length

            offset = (
                total_rows - row_index - 0.5
            ) * y_offset

            axis.plot(
                np.arange(segment_start, segment_end),
                ecg[lead_name]
                .to_numpy()[segment_start:segment_end]
                + offset,
                linewidth=0.6,
                color="black",
            )

            add_lead_label(
                axis,
                segment_start + 5,
                offset - 0.25,
                lead_name,
            )

    # Full-width rhythm strips.
    for strip_index, lead_name in enumerate(rhythm_strips):
        if lead_name not in ecg.columns:
            continue

        offset = (
            total_rows - (3 + strip_index) - 0.5
        ) * y_offset

        axis.plot(
            np.arange(total_samples),
            ecg[lead_name].to_numpy() + offset,
            linewidth=0.6,
            color="black",
        )

        add_lead_label(
            axis,
            5,
            offset - 0.25,
            lead_name,
        )

    figure.text(
        0.02,
        0.02,
        (
            f"{DEFAULT_MM_PER_SECOND:g} mm/s    "
            f"{DEFAULT_MM_PER_MV:g} mm/mV    "
            f"{sampling_rate:g} Hz"
        ),
        weight="bold",
        fontsize=9.5,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    args = parse_arguments()

    if not args.matched_csv.exists():
        raise FileNotFoundError(
            f"Matched cohort CSV does not exist: {args.matched_csv}"
        )

    if not args.ecg_root.exists():
        raise FileNotFoundError(
            f"ECG root directory does not exist: {args.ecg_root}"
        )

    matched = pd.read_csv(
        args.matched_csv,
        usecols=["subject_id", "study_id"],
    )

    matched = matched.dropna(
        subset=["subject_id", "study_id"]
    ).copy()

    matched["subject_id"] = matched["subject_id"].astype("int64")
    matched["study_id"] = matched["study_id"].astype("int64")

    # A study may appear more than once if it matches multiple laboratory
    # measurements. The ECG image only needs to be generated once.
    studies = matched.drop_duplicates(
        subset=["subject_id", "study_id"]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_count = 0
    missing_count = 0
    failed_count = 0
    skipped_count = 0

    for row_number, row in enumerate(
        studies.itertuples(index=False),
        start=1,
    ):
        subject_id = int(row.subject_id)
        study_id = int(row.study_id)

        subject_output_dir = args.output_dir / str(subject_id)
        output_image = subject_output_dir / f"{study_id}.png"

        if output_image.exists() and not args.overwrite:
            skipped_count += 1
            continue

        record_path = locate_record(
            ecg_root=args.ecg_root,
            subject_id=subject_id,
            study_id=study_id,
        )

        if record_path is None:
            missing_count += 1
            print(
                f"Missing WFDB record: subject_id={subject_id}, "
                f"study_id={study_id}"
            )
            continue

        try:
            ecg, record_sampling_rate = read_wfdb_record(record_path)

            sampling_rate = (
                record_sampling_rate
                if record_sampling_rate > 0
                else args.sampling_rate
            )

            corrected_ecg = apply_baseline_correction(
                ecg=ecg,
                sampling_rate=sampling_rate,
                cutoff_hz=args.baseline_cutoff,
            )

            plot_ecg(
                ecg=corrected_ecg,
                sampling_rate=sampling_rate,
                output_path=output_image,
                dpi=args.dpi,
            )

            success_count += 1

        except Exception as error:
            failed_count += 1
            print(
                f"Failed: subject_id={subject_id}, "
                f"study_id={study_id}, error={error}"
            )

        if row_number % 100 == 0:
            print(
                f"Processed {row_number:,}/{len(studies):,} studies."
            )

    print()
    print("Image-generation summary")
    print("------------------------")
    print(f"Unique ECG studies: {len(studies):,}")
    print(f"Successfully generated: {success_count:,}")
    print(f"Already existed: {skipped_count:,}")
    print(f"Missing WFDB records: {missing_count:,}")
    print(f"Failed: {failed_count:,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
