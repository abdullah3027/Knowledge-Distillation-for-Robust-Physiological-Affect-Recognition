MuSe-Physio — Data Engineering & EDA

This folder contains the full preprocessing pipeline and exploratory analysis for the
MuSe 2021 Physiological Arousal Sub-Challenge. The goal is to turn raw per-subject
CSV signals into clean, windowed, normalised data ready for a classifier.



## What's in here


Data_Pipeline.py          
preprocessed_data/
EDA.ipynb                 ← full exploratory analysis notebook 

preprocessed_data/
  train.csv               ← 4,105 windows, 28 columns
  devel.csv               ← 1,413 windows, 28 columns
  test.csv                ← 1,408 windows, 28 columns (labels are NaN — withheld)
  norm_stats.csv          ← mean and std per feature, computed from train only
  class_thresholds.csv    ← p33 and p66 tertile boundaries with references


 Running the pipeline

Make sure the raw dataset is in the expected location:


D:/ovgu_folder/advance_topics_of_kmd/Data/
  MuSe-Physio Multimodal Physiological-Arousal (MuSe2021)/
    c4_muse_physio/c4_muse_physio/
      feature_segments/   ← BPM/, ECG/, resp/ subfolders
      label_segments/anno12_EDA/
      metadata/partition.csv

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
