# Gait as a Window into the Brain

**How much does the way an older adult walks tell us about their cognitive health, and does a foot sensor add anything to a simple clinical walking test?**

This repository contains a single, fully commented notebook that tests these questions on the public **GSTRIDE** dataset of 163 older adults. Every step explains what is happening and why, so you do not need a machine learning background to follow it.

The analysis supports the paper *"Psychointelligence for Dementia and Falls in Older Adults: Interpretable Cognitive-Decline Screening from Gait Sensor Data"*, accepted at **ICOIN 2027**. The notebook includes every extra check the reviewers asked for.

---

## Contents

1. [The idea in plain English](#1-the-idea-in-plain-english)
2. [What this project actually does](#2-what-this-project-actually-does)
3. [Results at a glance](#3-results-at-a-glance)
4. [Quick start](#4-quick-start)
5. [The data](#5-the-data)
6. [Glossary for beginners](#6-glossary-for-beginners)
7. [Walkthrough of the notebook](#7-walkthrough-of-the-notebook)
8. [What the model learned](#8-what-the-model-learned)
9. [The main findings](#9-the-main-findings)
10. [Honest limitations](#10-honest-limitations)
11. [Roadmap](#11-roadmap)
12. [Requirements](#12-requirements)
13. [Citation and credit](#13-citation-and-credit)

---

## 1. The idea in plain English

Walking feels automatic, but it is not. Staying upright and steady takes attention, planning and balance, and all three are jobs the brain does. So when the brain struggles, the way a person moves can change too.

This project tests how strongly movement is linked to cognitive decline, and which movement measurements carry that link.

The project sits under a research theme the CIoTh lab calls **Psychointelligence**: the idea that mental state leaves measurable traces in ordinary behaviour, and that everyday sensors can read those traces in a way people can understand and trust. This study tests two parts of that idea: physical mobility and cognition.

---

## 2. What this project actually does

The notebook takes a small set of walking measurements from a foot-mounted sensor, plus the Timed Up and Go (TUG) test, age and BMI, and tries to predict three yes or no questions about each person:

| Task | What it means | How it is defined in the data |
|---|---|---|
| **Cognitive decline** | Signs of cognitive decline (main task) | GDS score of 3 or more |
| **Falls** | Fell at least once in the last year | Self-reported yes or no |
| **Frailty** | Physically frail | Fried score of 2 or more |

Three design choices matter more than the accuracy numbers:

- **Explainability first.** The main model is plain logistic regression, chosen in advance, so you can read exactly which measurements pushed a prediction up or down.
- **Redundancy removed before modelling.** Walking speed, stride length and cadence are largely the same information in three costumes. The notebook measures the overlap and keeps only clean representatives.
- **Honest testing.** Every score comes with a 95% confidence interval, and the notebook checks for data leakage, uses nested cross-validation, and tests whether the foot sensor adds anything beyond the TUG test.

---

## 3. Results at a glance

Scores are **ROC-AUC** from out-of-fold predictions, using 5-fold cross-validation repeated 10 times. 0.5 means the model is guessing. 1.0 is perfect.

**Cognitive decline (main task)**

| Measure | Result |
|---|---|
| AUC, logistic regression (chosen in advance) | **0.747** (95% CI 0.660 to 0.828) |
| AUC, nested cross-validation (stricter check) | **0.710** (95% CI 0.601 to 0.801) |
| At threshold 0.5: cases caught | **28 of 43** (sensitivity 0.66) |
| Specificity / precision | 0.72 / 0.46 |
| Share of people referred | 38% |

**What carries the signal?** (ablation)

| Measurements used | AUC (95% CI) |
|---|---|
| Age + BMI only | 0.659 (0.555 to 0.754) |
| Walking speed only | 0.656 (0.569 to 0.741) |
| Sensor gait only (speed, clearance, stride-time variability) | 0.664 (0.568 to 0.758) |
| **TUG time only** | **0.729** (0.643 to 0.808) |
| TUG + sensor gait | 0.726 (0.636 to 0.809) |
| Full model (6 features) | 0.747 (0.660 to 0.828) |

Three things worth noticing:

- **TUG time does most of the work.** On its own it almost matches the full model. Adding the foot-sensor features to TUG does not clearly improve the result.
- **The model finds moderate or worse decline, not mild decline.** AUC is 0.802 for GDS 4 and above, but only 0.619 for GDS 3 (not reliably better than guessing).
- **Simple beats complex here.** In nested cross-validation, logistic regression was chosen in 29 of 50 folds, ahead of random forest (14) and gradient boosting (7).

> An earlier version of this notebook reported AUC 0.752 and 31 of 43 cases caught. Those came from fold-averaged AUC and a single data split. The numbers above are the more reliable, repeated estimates.

---

## 4. Quick start

### Option A: Google Colab (easiest, nothing to install)

1. Open the notebook in Colab.
2. Download `data/Database_register.xlsx` from this repository (it is included, see [The data](#5-the-data)).
3. Upload it to the Colab session using the folder icon on the left, so it lands at `/content/Database_register.xlsx`.
4. Run all cells from top to bottom.

### Option B: On your own machine

```bash
git clone https://github.com/CollinsLemeke/Detecting-Cognitive-Decline-Falls-and-Frailty.git
cd Detecting-Cognitive-Decline-Falls-and-Frailty
pip install -r requirements.txt
jupyter notebook notebooks/gstride_psychointelligence.ipynb
```

The data file is already in `data/Database_register.xlsx`, so there is nothing else to download.

### Option C: Kaggle

The notebook also runs on Kaggle with the GSTRIDE dataset attached. A full run takes about 5 to 10 minutes on CPU (nested cross-validation is the slowest step). It does not use a GPU.

Repository layout:

```
.
├── data/
│   └── Database_register.xlsx      # GSTRIDE register, CC BY 4.0 (see The data)
├── notebooks/
│   └── gstride_psychointelligence.ipynb
├── src/
│   └── reproduce.py                # script version of the original analysis
├── CITATION.cff
├── requirements.txt
└── README.md
```

---

## 5. The data

**GSTRIDE**, a public dataset of 163 older adults. Each person was assessed clinically and then walked while wearing a single inertial sensor on the foot, which recorded spatio-temporal gait parameters such as stride length, stride time, speed, foot angles and clearance. The dataset also includes the standard clinical assessments used here (GDS, Fried phenotype and the TUG test).

Reported characteristics of the cohort: **163 participants, mean age 82.6 years (SD 6.2), roughly 72% women**. The notebook loads 163 rows and 60 columns.

Source paper:

> García-de-Villa, S., Jiménez, A. R., del-Ama, A. J., et al. (2023). *A database with frailty, functional and inertial gait metrics for the research of fall causes in older adults.* **Scientific Data**, 10, 566. https://doi.org/10.1038/s41597-023-02428-0

**Licence and redistribution.** GSTRIDE is published on Zenodo under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence, which allows redistribution as long as the creators are credited. The participant register used here (`data/Database_register.xlsx`) is included in this repository under that licence for convenience. The full dataset, including the raw inertial recordings, is available from the original record: https://doi.org/10.5281/zenodo.6883292

If you reuse the data, please cite the authors as shown in [Citation and credit](#13-citation-and-credit). The MIT licence in this repository covers the code and analysis only, not the dataset.

---

## 6. Glossary for beginners

| Term | What it means in plain English |
|---|---|
| **IMU** | Inertial Measurement Unit. A small motion sensor, like the one in a phone or fitness tracker, worn on the foot here. |
| **GDS** | Global Deterioration Scale. A clinical score for stages of cognitive decline, from 1 (none) to 7 (severe). A score of 3 or more is used here as "signs of decline". It is not a dementia diagnosis. |
| **Fried phenotype** | A standard five-part checklist for physical frailty. A score of 2 or more is used here as "frail". |
| **TUG** | Timed Up and Go. A simple clinical test: stand up from a chair, walk 3 metres, turn, come back, sit down. Longer time means worse mobility. |
| **Clearance** | How high the foot lifts off the ground during a step. Low clearance means shuffling. |
| **Stride time variability** | How inconsistent the rhythm of walking is from stride to stride. |
| **Feature** | One input measurement the model is allowed to use. |
| **Target** | The yes or no answer the model is trying to predict. |
| **Class imbalance** | When one answer is much rarer than the other. Rarer means fewer examples to learn from, so harder. |
| **VIF** | Variance Inflation Factor, a redundancy meter. Above 5 is a warning, above 10 is severe. |
| **Imputation** | Filling in a missing value. Here, with the column's median (its middle value). |
| **Cross-validation** | Splitting people into groups, training on most and testing on the rest, then rotating so everyone gets a turn as the test set. |
| **Nested cross-validation** | Cross-validation inside cross-validation. The inner loop picks the algorithm; the outer loop tests it on people it has never seen. It stops the model-choosing step from flattering the score. |
| **Data leakage** | When information from the test people sneaks into training, making results look better than they are. |
| **ROC-AUC** | One number summarising how well the model separates the two groups. 0.5 is guessing, 1.0 is perfect. |
| **Confidence interval (CI)** | A range that shows how sure we are about a number. A 95% CI that is narrow means the result is precise. |
| **Bootstrap** | Re-sampling the people many times to see how much a result would change by chance. Used here to build confidence intervals. |
| **Ablation** | Removing parts of the model to see which parts actually matter. |
| **Sensitivity / specificity / precision** | Share of real cases caught / share of healthy people correctly cleared / share of referrals that are real cases. |
| **Calibration** | Whether a predicted risk of 40% really means about 40% of those people have the condition. |
| **Odds ratio (OR)** | How much the odds of a condition change when a measurement goes up by one standard deviation (SD). Above 1 raises the odds, below 1 lowers them. |
| **Kappa / phi** | Measures of how much two yes or no labels agree, beyond chance. |

---

## 7. Walkthrough of the notebook

The notebook runs in 14 main steps. Steps marked "b", "c" or "d", and the revised steps, were added to answer the ICOIN reviewers.

**Step 0. Set up the tools.** Loads the libraries, defines a consistent chart style, and defines the helper tools used later (repeated cross-validation, confidence intervals, in-fold VIF).

**Step 1. Load the data.** Reads the real column names from row 2 and the records from row 4 onwards. Result: 163 people, 60 columns.

**Step 2. First look.** A doughnut chart showing how much of the raw table is filled in.

**Step 3. Clean the data.** Fixes the fall label (`"NO"` vs `"NO "`), converts age bands to midpoints, and builds the three targets.

**Step 4. Visualise the targets.**

| Target | People with the condition | Share |
|---|---|---|
| Cognitive decline | 43 of 163 | 26% |
| Falls | 86 of 163 | 53% |
| Frailty | 49 of 163 | 30% |

**Step 5. Who is in the study.** Mostly women, mostly aged 80 and over.

**Step 6. Choose features and remove overlaps.** A correlation heatmap and VIF scores show that speed, stride length, cadence, clearance variability and swing carry overlapping information. Final feature set of six: `age`, `bmi`, `tug`, `step_speed`, `clearance`, `stridetime_std`.

**Step 6b. Leakage check (new).** The VIF selection is repeated inside every training fold, so test people never influence the choice. Result: AUC 0.746 vs 0.747, difference 0.000. The original feature choice did not inflate the results.

**Step 7. Handle missing values.** Only TUG has gaps, filled with the median inside the modelling pipeline.

**Step 8. See the signal with your own eyes.** Box plots of TUG time and walking speed for each group.

**Step 9. How training and testing work.** 5-fold cross-validation repeated 10 times.

**Step 10. Train and compare.** Logistic regression, random forest and gradient boosting on all three tasks.

**Step 10b. Nested cross-validation (new).** Algorithm choice moves inside an inner loop. Result: AUC 0.710, with logistic regression chosen most often.

**Step 11. Performance, revised.** ROC curves for all 10 repeats, the confusion matrix at a threshold of 0.5 fixed in advance (28 of 43 caught), and the trade-off at other thresholds.

**Step 11b. Ablation (new).** Is the foot sensor worth it? TUG alone vs sensor alone vs full model.

**Step 11c. Calibration (new).** Brier score, calibration slope and plot.

**Step 11d. Results by GDS stage (new).** Does the model find mild cases, or only more advanced ones?

**Step 12. What the model learned, revised.** Odds ratios per SD with 95% confidence intervals and a bootstrap check.

**Step 13. How the three tasks relate, revised.** Compares the model risk scores and the real labels across tasks, and tests whether gait adds information beyond frailty, falls and age.

**Step 14. Findings and limits.**

---

## 8. What the model learned

Odds ratios from logistic regression on the cognitive decline task, per one standard deviation (SD) of each measurement, with 95% confidence intervals:

| Feature | 1 SD equals | Odds ratio (95% CI) | Clear effect? |
|---|---|---|---|
| TUG time | 9.4 seconds | **2.34** (1.31 to 4.18) | **Yes, raises the odds** |
| BMI | 5.0 kg/m² | **0.52** (0.33 to 0.80) | **Yes, lowers the odds** |
| Age | 6.4 years | 1.45 (0.92 to 2.28) | No |
| Step speed | 0.31 | 0.88 (0.41 to 1.86) | No |
| Stride time variability | 0.06 | 0.69 (0.38 to 1.26) | No |
| Foot clearance | 0.14 | 0.63 (0.34 to 1.17) | No |

**How to read this.** A slower Timed Up and Go is the only clear risk signal. Lower BMI is also linked to higher odds, which may reflect the weight loss that often comes with cognitive decline, but this data cannot show cause. The sensor measures (speed, clearance, variability) show **no clear independent effect** once TUG and BMI are in the model. Their confidence intervals all include 1.

Bootstrap confidence intervals (1,000 resamples) give the same conclusions.

---

## 9. The main findings

**1. TUG carries most of the signal.** TUG time alone reaches AUC 0.729, close to the full model (0.747). The foot sensor adds little on top of it in this cohort. TUG is a compound task (standing up, walking, turning, sitting down), which may make it more mentally demanding than walking alone.

**2. The signal is strongest for more advanced decline.** The model separates GDS 4 and above well (AUC 0.802) but not mild decline at GDS 3 (AUC 0.619). It is not an early-detection tool.

**3. Falls and frailty are related, but not the same.** The falls and frailty risk scores correlate at 0.93, but the real labels agree only moderately (phi 0.49, kappa 0.44). The high score correlation mostly reflects that both models use the same inputs.

**4. TUG reflects more than frailty.** With frailty, falls and age held fixed, slower TUG is still linked to cognitive decline (adjusted OR 2.22, 95% CI 1.28 to 3.86). Adding TUG, gait and BMI to a frailty + falls + age model raises AUC from 0.654 to 0.738 (gain +0.083, 95% CI −0.005 to +0.172), which is suggestive but not conclusive.

**In one sentence:** a compound mobility test carries a measurable cognitive signal that is not explained by frailty alone, but in this dataset the foot sensor adds little beyond that test.

---

## 10. Honest limitations

Please read this section before citing anything above.

- **This is a research signal, not a diagnosis.** GDS 3 or more is a staging threshold, not a dementia diagnosis. Nothing here should be used to make a clinical decision about a real person.
- **Not an early-detection tool.** Mild decline (GDS 3) is not reliably detected.
- **Small sample.** 163 people, of whom 43 show signs of decline.
- **Narrow cohort.** Mostly women, mostly aged 80 and over, from one study, with no external validation.
- **Snapshot, not a prediction.** Walking and cognition were measured at the same time, so this cannot show that walking predicts *future* decline.
- **Risk scores are not probabilities.** Because of class-balanced weighting, predicted risks are too high (mean 0.44 vs a true rate of 0.26). Use them for ranking only.
- **Confounders were not controlled.** Pain, medication, walking aids and other conditions can all affect gait.
- **Odds ratio intervals are exploratory.** The odds ratio model was fitted on all 163 people, and its intervals do not account for the feature-selection step.

---

## 11. Roadmap

Done in this version:

- ✅ Leakage check with feature selection inside each fold
- ✅ Nested cross-validation
- ✅ Bootstrap confidence intervals on every score
- ✅ Threshold fixed in advance, with the trade-off at other thresholds shown
- ✅ Calibration check
- ✅ Ablation against TUG alone
- ✅ Adjusted analysis beyond frailty and falls

Still to do:

1. **External validation** on a second, independent cohort.
2. **Longitudinal testing**: does gait today predict cognitive decline later?
3. **Recalibration**, so the risk scores can be read as real probabilities.
4. **Richer protocols**: compare TUG with dual-task walking and free-living sensor data.
5. **Subgroup checks** by sex, age band and mobility aids.
6. **More psychointelligence layers**: mood, fear of falling, daily routine and home environment.

---

## 12. Requirements

Python 3.9 or newer.

```
pandas
numpy
scipy
matplotlib
scikit-learn
statsmodels
openpyxl
```

`openpyxl` is required to read the `.xlsx` data file. Everything except `statsmodels` and `openpyxl` is preinstalled in Google Colab, and both install in seconds.

The random seed is fixed at 42, so re-running the notebook on the same data reproduces the same numbers.

---

## 13. Citation and credit

**Paper.** C. Iwendi, C. Lemeke, M. Raji, G. O. Oghenekioja, M. D. Edom and S. A. Afolaranmi, "Psychointelligence for Dementia and Falls in Older Adults: Interpretable Cognitive-Decline Screening from Gait Sensor Data," accepted at the *International Conference on Information Networking (ICOIN 2027)*.

**Dataset.** Please cite the original GSTRIDE data descriptor:

> García-de-Villa, S., Jiménez, A. R., del-Ama, A. J., et al. (2023). A database with frailty, functional and inertial gait metrics for the research of fall causes in older adults. *Scientific Data*, 10, 566. https://doi.org/10.1038/s41597-023-02428-0

**This analysis.** Collins Lemeke, Centre of Intelligence of Things (CIoTh), University of Greater Manchester.

**Contact and links.** GitHub [@CollinsLemeke](https://github.com/CollinsLemeke) · Kaggle [collinslemeke](https://www.kaggle.com/collinslemeke)

---

### A note on intended use

This is a research notebook built for learning and for methodological transparency. It is not a medical device, it has not been clinically validated, and it must not be used to screen, diagnose or advise any real person. If you are worried about your own memory or mobility, or someone else's, please speak to a doctor.
