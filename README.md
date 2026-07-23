# mimic-iv-ecg-ntprobnp-reproducibility

This repository contains the reproducibility code used to construct the MIMIC-IV external validation cohort and preprocess electrocardiogram (ECG) data for an artificial intelligence (AI)-enabled single-lead (Lead I) ECG model for predicting N-terminal pro-B-type natriuretic peptide (NT-proBNP).

## Repository Overview

This repository includes code for:

1. Identifying NT-proBNP laboratory measurements from the MIMIC-IV database.
2. Matching ECGs and NT-proBNP measurements from the same patient and hospital admission.
3. Retaining ECG–NT-proBNP pairs with an absolute time difference of ≤24 hours.
4. Generating standardized 12-lead ECG images from MIMIC-IV-ECG WFDB waveform records.
5. Extracting the single-lead ECG (Lead I) used as input to the AI model.

## Cohort Definition

Patients were included if they had:

- At least one NT-proBNP laboratory measurement (MIMIC-IV `itemid = 50963`).
- An ECG from the same patient and hospital admission.
- An ECG acquired within ±24 hours of the NT-proBNP measurement.

All eligible ECG–NT-proBNP pairs satisfying these criteria were retained.

## Data Requirements

Access to the following credentialed datasets is required:

- MIMIC-IV
- MIMIC-IV-ECG Diagnostic Electrocardiogram Matched Subset

These datasets are **not** distributed with this repository.

Expected input files include:

- `labevents.csv`
- `records_w_diag_icd10.csv`
- MIMIC-IV-ECG WFDB waveform files

The ECG linkage file is expected to contain:

- `subject_id`
- `study_id`
- `ecg_time`
- `hosp_hadm_id`

## Repository Structure

### `01_match_ecg_ntprobnp.py`

Identifies NT-proBNP laboratory measurements (`itemid = 50963`) and matches them to ECGs from the same patient and hospital admission. Only ECG–NT-proBNP pairs with an absolute time difference of ≤24 hours are retained.

### `02_generate_ecg_images.py`

Reads the matched cohort, loads the corresponding MIMIC-IV-ECG WFDB waveform records, performs baseline correction, and generates standardized 12-lead ECG images.

### `03_extract_lead1_from_pdf.py`

Extracts the single-lead ECG (Lead I) from the standardized ECG images using predefined crop coordinates to reproduce the model input used in the study.

The crop coordinates are specific to the ECG image layout used in this workflow and may require modification for different ECG formats or image layouts.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Step 1: ECG–NT-proBNP Matching

```bash
python 01_match_ecg_ntprobnp.py \
    --labevents /path/to/mimiciv/hosp/labevents.csv \
    --ecg-linkage /path/to/records_w_diag_icd10.csv \
    --output /path/to/ntprobnp_ecg_within24h.csv
```

The output contains all ECG–NT-proBNP pairs from the same patient and hospital admission with an absolute time difference of ≤24 hours.

## Step 2: Generate Standardized ECG Images

```bash
python 02_generate_ecg_images.py \
    --matched-csv /path/to/ntprobnp_ecg_within24h.csv \
    --ecg-root /path/to/mimic-iv-ecg/files \
    --output-dir /path/to/ECG_Images_24h
```

Generated images are organized as:

```text
ECG_Images_24h/
└── subject_id/
    └── study_id.png
```

## Step 3: Extract Single-Lead ECG (Lead I)

```bash
python 03_extract_lead1_from_pdf.py \
    --input-csv /path/to/ecg_image_paths.csv \
    --path-column image_path \
    --output-dir /path/to/lead1_images \
    --left 82 \
    --top 458 \
    --right 577 \
    --bottom 585 \
    --dpi 200
```

This step extracts the Lead I region used as input to the AI model. The predefined crop coordinates correspond to the ECG image layout generated in this workflow.

## Important Notes

- ECGs and NT-proBNP measurements are matched only when they belong to the same patient and hospital admission.
- The matching criterion is an absolute ECG–NT-proBNP time difference of ≤24 hours.
- All eligible ECG–NT-proBNP pairs are retained.
- Institutional file paths and protected clinical data are intentionally excluded from this repository.
- Users are responsible for obtaining access to MIMIC-IV and complying with all PhysioNet data use agreements.

## Citation

If you use this code in your research, please cite the associated manuscript:

> **Alkhaleefah M, et al.** *Deep Learning-Based BNP and NT-proBNP Estimation from Wearable-Compatible
Single-Lead ECG Images for Heart Failure Assessment.* 
