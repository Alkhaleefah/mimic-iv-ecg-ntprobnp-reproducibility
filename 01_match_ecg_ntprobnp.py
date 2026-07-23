#!/usr/bin/env python3
"""
Match MIMIC-IV ECG records to NT-proBNP measurements.

Eligibility criteria
--------------------
An ECG–NT-proBNP pair is retained when:

1. The laboratory event has itemid 50963.
2. The ECG and laboratory measurement belong to the same subject.
3. The ECG and laboratory measurement belong to the same hospital admission.
4. The absolute time difference between ECG acquisition and laboratory
   measurement is no greater than the specified matching window.

By default, the matching window is 24 hours.

All eligible ECG–NT-proBNP pairs are retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NTPROBNP_ITEMID = 50963


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match ECG records to NT-proBNP measurements."
    )

    parser.add_argument(
        "--labevents",
        required=True,
        type=Path,
        help="Path to the MIMIC-IV labevents.csv file.",
    )

    parser.add_argument(
        "--ecg-linkage",
        required=True,
        type=Path,
        help=(
            "CSV containing subject_id, study_id, ecg_time, and "
            "hosp_hadm_id."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the matched output CSV.",
    )

    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Maximum absolute ECG–laboratory time difference in hours.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Number of labevents rows read per chunk.",
    )

    return parser.parse_args()


def validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_ntprobnp_events(
    labevents_path: Path,
    chunksize: int,
) -> pd.DataFrame:
    """
    Load only NT-proBNP laboratory events from labevents.csv.

    Chunked reading is used because the MIMIC-IV laboratory table can be large.
    """
    required_columns = [
        "subject_id",
        "hadm_id",
        "charttime",
        "itemid",
        "valuenum",
        "valueuom",
    ]

    selected_chunks: list[pd.DataFrame] = []

    print("Reading NT-proBNP measurements from labevents.csv...")

    reader = pd.read_csv(
        labevents_path,
        usecols=required_columns,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        ntprobnp_chunk = chunk.loc[
            chunk["itemid"].eq(NTPROBNP_ITEMID)
        ].copy()

        if not ntprobnp_chunk.empty:
            selected_chunks.append(ntprobnp_chunk)

        if chunk_number % 10 == 0:
            print(f"Processed {chunk_number:,} chunks.")

    if not selected_chunks:
        raise ValueError(
            f"No laboratory records were found for itemid "
            f"{NTPROBNP_ITEMID}."
        )

    labs = pd.concat(selected_chunks, ignore_index=True)

    labs["charttime"] = pd.to_datetime(
        labs["charttime"],
        errors="coerce",
    )

    labs["valuenum"] = pd.to_numeric(
        labs["valuenum"],
        errors="coerce",
    )

    # Matching by hospital admission requires a nonmissing hadm_id.
    labs = labs.dropna(
        subset=["subject_id", "hadm_id", "charttime", "valuenum"]
    ).copy()

    labs["subject_id"] = labs["subject_id"].astype("int64")
    labs["hadm_id"] = labs["hadm_id"].astype("int64")

    print(f"NT-proBNP laboratory records: {len(labs):,}")
    print(
        "Patients with NT-proBNP measurements: "
        f"{labs['subject_id'].nunique():,}"
    )
    print(
        "Admissions with NT-proBNP measurements: "
        f"{labs['hadm_id'].nunique():,}"
    )

    return labs


def load_ecg_linkage(ecg_linkage_path: Path) -> pd.DataFrame:
    """Load ECG metadata containing hospital-admission linkage."""
    required_columns = [
        "subject_id",
        "study_id",
        "ecg_time",
        "hosp_hadm_id",
    ]

    ecg = pd.read_csv(
        ecg_linkage_path,
        usecols=required_columns,
        low_memory=False,
    )

    ecg["ecg_time"] = pd.to_datetime(
        ecg["ecg_time"],
        errors="coerce",
    )

    ecg = ecg.dropna(
        subset=[
            "subject_id",
            "study_id",
            "ecg_time",
            "hosp_hadm_id",
        ]
    ).copy()

    ecg["subject_id"] = ecg["subject_id"].astype("int64")
    ecg["study_id"] = ecg["study_id"].astype("int64")
    ecg["hosp_hadm_id"] = ecg["hosp_hadm_id"].astype("int64")

    print(f"ECG records with admission linkage: {len(ecg):,}")
    print(f"Patients with ECG records: {ecg['subject_id'].nunique():,}")

    return ecg


def match_ecg_to_ntprobnp(
    labs: pd.DataFrame,
    ecg: pd.DataFrame,
    window_hours: float,
) -> pd.DataFrame:
    """
    Match ECGs and NT-proBNP measurements by patient and admission.

    All candidate combinations from the same subject and admission are first
    created. Pairs outside the absolute time window are subsequently removed.
    """
    if window_hours <= 0:
        raise ValueError("--window-hours must be greater than zero.")

    print("Merging ECGs and laboratory measurements by patient and admission...")

    matched = pd.merge(
        labs,
        ecg,
        left_on=["subject_id", "hadm_id"],
        right_on=["subject_id", "hosp_hadm_id"],
        how="inner",
        validate="many_to_many",
    )

    if matched.empty:
        raise ValueError(
            "No ECG and NT-proBNP records matched by subject and admission."
        )

    matched["time_difference"] = (
        matched["charttime"] - matched["ecg_time"]
    ).abs()

    matched["time_difference_hours"] = (
        matched["time_difference"].dt.total_seconds() / 3600.0
    )

    matched = matched.loc[
        matched["time_difference_hours"].le(window_hours)
    ].copy()

    matched = matched.sort_values(
        by=[
            "subject_id",
            "hadm_id",
            "study_id",
            "charttime",
        ]
    ).reset_index(drop=True)

    return matched


def print_summary(matched: pd.DataFrame, window_hours: float) -> None:
    print()
    print("Matching summary")
    print("----------------")
    print(f"Matching window: ±{window_hours:g} hours")
    print(f"Eligible ECG–NT-proBNP pairs: {len(matched):,}")
    print(f"Unique patients: {matched['subject_id'].nunique():,}")
    print(f"Unique admissions: {matched['hadm_id'].nunique():,}")
    print(f"Unique ECG studies: {matched['study_id'].nunique():,}")


def main() -> None:
    args = parse_arguments()

    validate_file(args.labevents, "Laboratory-events file")
    validate_file(args.ecg_linkage, "ECG-linkage file")

    labs = load_ntprobnp_events(
        labevents_path=args.labevents,
        chunksize=args.chunksize,
    )

    ecg = load_ecg_linkage(args.ecg_linkage)

    matched = match_ecg_to_ntprobnp(
        labs=labs,
        ecg=ecg,
        window_hours=args.window_hours,
    )

    if matched.empty:
        raise ValueError(
            "No ECG–NT-proBNP pairs satisfied the time-window criterion."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output, index=False)

    print_summary(matched, args.window_hours)
    print(f"\nSaved matched cohort to: {args.output}")


if __name__ == "__main__":
    main()
