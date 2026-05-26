# MuSe-Physio — Data Engineering & EDA

This folder contains the full preprocessing pipeline and exploratory analysis for the
**MuSe 2021 Physiological Arousal Sub-Challenge**. The goal is to turn raw per-subject
CSV signals into clean, windowed, normalised data ready for a classifier.

---

## What's in here

```
Data_Pipeline.py          ← run this first — produces everything in preprocessed_data/
generate_report.py        ← optional: re-generates Preprocessing_Report.docx
generate_eda_notebook.py  ← optional: re-generates EDA.ipynb from scratch

Preprocessing_Report.docx ← 2-page summary with pipeline diagrams (already generated)
EDA.ipynb                 ← full exploratory analysis notebook (already generated)

preprocessed_data/
  train.csv               ← 4,105 windows, 28 columns
  devel.csv               ← 1,413 windows, 28 columns
  test.csv                ← 1,408 windows, 28 columns (labels are NaN — withheld)
  norm_stats.csv          ← mean and std per feature, computed from train only
  class_thresholds.csv    ← p33 and p66 tertile boundaries with references
```

---

## Running the pipeline

Make sure the raw dataset is in the expected location:

```
D:/ovgu_folder/advance_topics_of_kmd/Data/
  MuSe-Physio Multimodal Physiological-Arousal (MuSe2021)/
    c4_muse_physio/c4_muse_physio/
      feature_segments/   ← BPM/, ECG/, resp/ subfolders
      label_segments/anno12_EDA/
      metadata/partition.csv
```

Then just run:

```bash
python Data_Pipeline.py
```

It will print each subject as it processes, show the window counts, print the
normalisation stats, print the arousal thresholds, and save all five output files
under `preprocessed_data/`. Takes under a minute on a normal machine.

---

## What the pipeline actually does (step by step)

**1. Load & merge per subject**
For each subject, it reads three separate CSVs — BPM, ECG, resp — and joins them
on `timestamp`. Then it attaches the EDA label from `anno12_EDA/`. Any subject
whose BPM file is missing gets skipped with a `[SKIP]` notice.

**2. Slide a 3-second window**
The data runs at 2 Hz (one row every 500 ms), so a 3-second window is 6 rows.
Windows are non-overlapping — the next window starts right where the last one ended.
No data leakage between windows, and no redundant rows.

**3. Aggregate each window**
For each of the three signals, three statistics are computed per window:
mean, median, and mode. Same thing is done for the EDA label. That gives
9 feature columns and 3 label columns per window, plus 4 ID columns
(subject_id, window_id, start_timestamp, end_timestamp).

**4. Normalise — train set only**
Z-score normalisation is computed from the train partition mean and standard
deviation. The same mu/sigma is then applied to devel and test. This is the
correct way to do it — if you fit the scaler on devel or test data you're leaking
information. The raw (unnormalised) columns are kept alongside the `_norm` ones
so you can choose which to feed your model.

**5. Classify into Low / Medium / High arousal**
Thresholds are computed as the 33rd and 66th percentiles of `label_mean` in the
train set only. Every window in all three splits then gets a class label based on
those thresholds. This produces three class columns: `class_mean`, `class_median`,
`class_mode`. The resulting class distribution is nearly perfectly balanced
(~33% each) which is exactly what you want.

---

## Output column layout (28 columns)

| Group | Columns |
|---|---|
| Identifiers | `subject_id`, `window_id`, `start_timestamp`, `end_timestamp` |
| Raw features | `BPM_mean/median/mode`, `ECG_mean/median/mode`, `resp_mean/median/mode` |
| Normalised features | same names with `_norm` suffix (Z-score, train params) |
| Labels | `label_mean`, `label_median`, `label_mode` |
| Arousal class | `class_mean`, `class_median`, `class_mode` (Low / Medium / High) |

For the test split, all label and class columns are NaN because the ground truth is
withheld. This is by design — don't try to fill them in.

---

## The arousal classification — why tertile split?

Short answer: because the EDA labels are continuous and normalised, not discrete scores
on a fixed scale like DEAP or MAHNOB-HCI. A fixed midpoint (e.g., 5.0 on a 1-9 scale)
would be meaningless here. A median split would throw away the middle-arousal region.

The tertile approach is validated by two papers:
- Wiem & Lachiri, IJACSA 2017 (MAHNOB-HCI + EDA)
- Sanchez-Reolid et al., Sensors 2022 (wrist EDA sensors)

Both show data-driven tertile splits outperform fixed boundaries for normalised
physiological arousal labels. The thresholds and references are stored in
`preprocessed_data/class_thresholds.csv` so they're reproducible.

---

## Exploring the data

Open `EDA.ipynb` in Jupyter. The notebook runs through 12 sections:

1. Setup & data loading
2. Dataset overview (dtypes, null checks, head)
3. Raw feature distributions (histograms, boxplots)
4. Label distribution with threshold lines
5. Arousal class balance per split
6. Normalised feature distributions (violin plots)
7. Correlation heatmap and feature-label correlations
8. Feature distributions broken down by arousal class (with ANOVA significance)
9. Subject-level analysis (windows per subject, per-subject EDA trajectory)
10. Train vs devel comparison (KDE overlap, KS test)
11. Outlier detection (|z| > 3 flagging)
12. Key findings summary table

If you need to regenerate it from scratch: `python generate_eda_notebook.py`

---

## Re-generating the report

```bash
python generate_report.py
```

This overwrites `Preprocessing_Report.docx` with a fresh 2-page Word document that
includes the pipeline diagram, key parameters table, classification rationale, and
the output file structure table. Useful if you changed any parameters and want the
report to reflect the new numbers.

---

## Key numbers to keep in mind

| Thing | Value |
|---|---|
| Subjects total | 69 (41 train / 14 devel / 14 test) |
| Sampling rate | 2 Hz (500 ms per row) |
| Window size | 3 seconds = 6 rows |
| Total windows | 6,926 (4,105 / 1,413 / 1,408) |
| Features | 9 raw + 9 normalised |
| Arousal thresholds | p33 = −0.2797, p66 = 0.0370 |
| Class balance (train) | Low 33.0% / Medium 33.0% / High 34.0% |

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scipy
python-docx
nbformat
```

Install with `pip install pandas numpy matplotlib seaborn scipy python-docx nbformat`.
