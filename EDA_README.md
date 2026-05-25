# EDA.ipynb — Exploratory Data Analysis Walkthrough

This notebook explores the preprocessed MuSe-Physio data before any modelling starts.
The idea is simple — before I train anything, I want to understand what the data actually
looks like, where the tricky parts are, and whether the preprocessing did what it was
supposed to do. Everything runs off the five files in `preprocessed_data/`.

---

## How to run it

Make sure you've already run `Data_Pipeline.py` so the `preprocessed_data/` folder exists.
Then just open `EDA.ipynb` in Jupyter and run all cells top to bottom. The notebook reads
from `preprocessed_data/` using a relative path, so run it from the project root.

```
preprocessed_data/
  train.csv
  devel.csv
  test.csv
  norm_stats.csv
  class_thresholds.csv
```

---

## Section-by-section walkthrough

### 1 · Setup & Data Loading

I import all the libraries here and set up the two colour palettes I use throughout:
- Blue/green/red for Low/Medium/High arousal classes
- Blue/green for train/devel split comparisons

Then I load all five output files and pull the two threshold values (p33 = −0.2797,
p66 = 0.0370) directly from `class_thresholds.csv`. I also define four column lists
— `RAW_FEATS`, `NORM_FEATS`, `LABEL_COLS`, `CLASS_COLS` — so I don't have to retype
them in every cell.

The final print confirms we have 4,105 / 1,413 / 1,408 windows across train / devel / test.

---

### 2 · Dataset Overview

**dtypes** — I check that every column loaded with the right type. The 4 ID columns
are int64, the 18 feature columns are float64, and the 3 class columns are object
(the strings "Low", "Medium", "High").

**Null check** — I wrote a small function that prints only the columns that actually
have missing values. Train and devel are fully clean. Test has 1,408 NaNs across all
6 label and class columns — that's expected, the challenge withholds the test ground truth.

**head()** — First 5 rows so I can see what real values look like. Subject 1, windows 0–4,
timestamps running 500ms to 15,000ms, BPM around 75–91, labels all falling in the High class.

**describe()** — Summary statistics (count, mean, std, min, quartiles, max) for all 9 raw
features on the train set. A few things stand out immediately: BPM runs from about 58 to
158 bpm with a mean of 99. ECG is nearly zero-centred (it's a voltage signal). Respiration
has the largest variance — resp_mode especially, with a std of nearly 2 and a min of −9.7.

---

### 3 · Raw Feature Distributions

A 3×3 grid of histograms — rows are BPM, ECG, resp; columns are mean, median, mode.
All on the train set. Each plot has a red dashed line at the column mean and a green
dotted line at the median, so I can immediately see skew.

BPM is roughly bell-shaped. ECG is tightly concentrated near zero with a long right tail
from a few high-voltage windows. Resp mean and median are symmetric, but resp_mode has
a heavy left tail — the mode within a window often falls at a negative respiration value.

---

### 4 · Label Distribution & Threshold Visualisation

This is the most important single plot in the notebook. I show the `label_mean`
distribution for both train (left) and devel (right) side by side. Each plot has:

- A histogram of the continuous EDA label
- A smooth KDE curve fitted over it
- Two vertical dashed lines at p33 (−0.2797) and p66 (0.0370)
- Shaded background regions: blue for Low, green for Medium, red for High

This confirms visually that the tertile split produces three balanced zones, that the
label is continuous and roughly bell-shaped (not discrete), and that devel's distribution
is slightly shifted from train — the devel subjects have slightly different arousal
baselines, which is normal when you split at the subject level.

---

### 5 · Arousal Class Distribution

Three bar charts — one per split. Each bar shows the window count and percentage for
Low, Medium, and High.

Train is almost perfectly balanced: 33% / 33% / 34%. That's exactly what the tertile
split is designed to produce. Devel skews a bit more toward High (38.5%) because the
thresholds were computed from train only — they don't align perfectly with devel's
distribution. Test shows NaN because labels are withheld.

---

### 6 · Normalised Feature Distributions

**Raw vs normalised comparison** — I use BPM_mean as the representative example.
Left panel shows the raw distribution, right panel shows the Z-score normalised version.
Both train and devel are overlaid so you can see that normalisation centres the data
at 0, and that train and devel still differ slightly — which is correct, since the
scaler was fit on train alone.

**Violin plot of all 9 normalised features** — All nine `_norm` columns in one chart,
with dashed lines at 0 (expected mean) and dotted lines at ±1 std. This is a good
sanity check: the train violins should all sit on 0. resp_mode has the widest violin
by far, which flags it as the most variable feature after normalisation.

---

### 7 · Correlation Analysis

**Heatmap** — Pearson correlation between all 9 raw features and `label_mean`, shown
as a lower-triangular matrix. The main takeaway: BPM_mean, BPM_median, BPM_mode are
nearly identical to each other (r > 0.96). Same for resp_mean and resp_median (r ~0.94).
ECG_mode is the odd one out — it only correlates ~0.47 with ECG_mean, meaning the mode
captures different information than the mean for ECG signals. Cross-signal correlations
are weak, which is good — the three physiological signals carry different information.

**Feature-label correlation bar chart** — Shows the Pearson r between each raw feature
and `label_mean` as a horizontal bar. Red = positive, blue = negative. All values are
very small (|r| < 0.15). No single feature linearly predicts arousal well. This is an
important finding — it tells me a linear model will struggle, and arousal is genuinely
a multivariate problem.

---

### 8 · Feature Distributions by Arousal Class

**Boxplots with ANOVA** — A 3×3 grid of boxplots, same layout as the histogram grid.
For each feature, I split the train windows by their arousal class (Low/Medium/High)
and draw a boxplot per class. I also run a one-way ANOVA automatically and add the
significance marker (ns / * / ** / ***) to the title. BPM shows some separation —
High arousal windows tend to have slightly higher heart rate — but the interquartile
ranges overlap heavily. Most features show ns or weak markers, confirming the features
don't cleanly separate the classes.

**Class means table** — Groups the train set by `class_mean` and computes the mean
of each raw feature per class. The numeric companion to the boxplots. For example:
High arousal has mean BPM ~101, Low has ~96 — a 5 bpm difference that sounds meaningful
but is buried inside a within-class std of ~20 bpm.

---

### 9 · Subject-Level Analysis

**Windows per subject** — Two bar charts (train, devel), one bar per subject, height
is the number of windows that subject contributes. Train subjects have 90–111 windows
each, devel 95–104. Very balanced — no subject is over- or under-represented.

**Subject-level mean EDA with error bars** — Each train subject gets one bar (their
mean `label_mean`) and one error bar (their std). Subjects are sorted left to right
by mean arousal. The p33 and p66 lines are overlaid. This is where inter-subject
variability becomes very visible: some subjects sit entirely below p33 (chronically
low arousal), others sit above p66 (chronically high). This is why cross-subject
generalisation is hard — arousal baselines are person-specific.

**Sample subjects time series** — For subjects 1, 8, and 42, I plot `label_mean`
window-by-window (i.e. across time during the session), with each point coloured
by its class. The threshold lines are overlaid. You can see how arousal evolves:
some subjects show a gradual drift in one direction, others flip between classes
rapidly within a single session.

---

### 10 · Train vs Devel Comparison

I run the two-sample Kolmogorov-Smirnov test on every raw feature, comparing the
train and devel distributions. The KS statistic measures the maximum gap between
the two empirical CDFs; the p-value says whether that gap is statistically significant.

Result: all 9 features are flagged DIFFERENT (p < 0.05). This is expected — train
and devel are different people, so their physiological distributions won't be identical.
It's a useful confirmation that the model can't just memorise training distributions
and expect them to hold at evaluation time.

---

### 11 · Outlier Detection

For each of the 9 normalised features, I count how many windows have an absolute Z-score
above 3 (more than 3 standard deviations from the train mean). BPM: zero outliers.
ECG and resp: around 1–2% of windows each. In total, 261 windows (6.36%) have at least
one feature outside ±3 sigma.

I don't remove these. Physiological extremes are real events — a spike in heart rate
or a very deep breath — and stripping them would distort the signal. The percentage
is low enough that they won't dominate any model.

---

### 12 · Key Findings Summary

**Markdown table** — A compact reference of 12 key facts from the entire analysis:
dataset size, window length, missing values, label range, class balance, normalisation
check, BPM/ECG/resp ranges, feature-label correlation, subject variability, and outlier rate.

**Summary DataFrame** — One row per split (Train/Devel/Test) showing subjects, total
windows, mean windows per subject, label mean and std, and class percentages. Good to
screenshot or include in a report.

---

## Key numbers at a glance

| | Train | Devel | Test |
|---|---|---|---|
| Subjects | 41 | 14 | 14 |
| Windows | 4,105 | 1,413 | 1,408 |
| Mean windows/subject | 100.1 | 100.9 | 100.6 |
| label_mean (mean) | −0.093 | −0.056 | withheld |
| % Low | 33.0% | 30.0% | withheld |
| % Medium | 33.0% | 31.5% | withheld |
| % High | 34.0% | 38.5% | withheld |

---

## What I found that matters for modelling

- No single feature predicts arousal linearly (|r| < 0.15 across the board) — need a multivariate approach
- BPM mean/median/mode are almost identical to each other — could drop two of the three without losing much
- ECG_mode behaves differently from ECG_mean/median — worth keeping separately
- resp_mode has very high variance — may need extra attention or could act as noise
- Inter-subject EDA baselines vary enormously — a model that learns subject-specific offsets will have an advantage
- Train and devel distributions are statistically different (KS test) — standard cross-subject generalisation challenge
- Outlier rate is low and confined to ECG and resp — no data quality issues
