# Gait as a Window into the Brain

**Can a machine spot early signs of cognitive decline just from the way an older adult walks?**

This repository contains a single, fully commented notebook that tests that question on the public **GSTRIDE** dataset of 163 older adults. Every step explains what is happening and why, so you do not need a machine learning background to follow it.

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
9. [The main finding](#9-the-main-finding)
10. [Honest limitations](#10-honest-limitations)
11. [Roadmap](#11-roadmap)
12. [Requirements](#12-requirements)
13. [Citation and credit](#13-citation-and-credit)

---

## 1. The idea in plain English

Walking feels automatic, but it is not. Staying upright and steady takes attention, planning and balance, and all three are jobs the brain does. So when the brain starts to struggle, the walk quietly changes before anyone notices a memory problem.

This project tests whether a machine can pick up that quiet change.

The project sits under a research theme the CIoTh lab calls **Psychointelligence**: the idea that mental state leaves measurable traces in ordinary behaviour, and that everyday sensors can read those traces in a way people can understand and trust.

---

## 2. What this project actually does

The notebook takes a small set of walking measurements from a foot-mounted sensor, plus age and BMI, and tries to predict three yes or no questions about each person:

| Task | What it means | How it is defined in the data |
|---|---|---|
| **Dementia** | Signs of cognitive decline | GDS score of 3 or more |
| **Falls** | Fell at least once in the last year | Self-reported yes or no |
| **Frailty** | Physically frail | Fried score of 2 or more |

Three different algorithms are trained on all three tasks, so you can see which style of model suits which problem. The whole thing runs in a few minutes on a free Google Colab session.

Two design choices matter more than the accuracy numbers:

- **Explainability first.** The headline model is plain logistic regression, so you can read exactly which measurements pushed a prediction up or down. A black box that scores slightly higher would be far less useful in a clinic.
- **Redundancy removed before modelling.** Walking speed, stride length and cadence are largely the same information in three costumes. Feeding all of them in makes the explanations meaningless. The notebook measures the overlap and keeps only clean representatives.

---

## 3. Results at a glance

Scores are **ROC-AUC**, averaged over 5-fold cross-validation repeated 10 times. 0.5 means the model is guessing. 1.0 is perfect. Around 0.75 is genuinely useful for screening.

| Task | Best algorithm | ROC-AUC |
|---|---|---|
| Dementia | Logistic Regression | **0.752** |
| Falls | Random Forest | **0.814** |
| Frailty | Logistic Regression | **0.749** |

On the dementia task, at the standard 0.5 decision threshold, the model **caught 31 of the 43 people showing signs of decline, a catch rate of 72%**.

Two things worth noticing:

- The **simple, explainable model wins** on the two rarer conditions. More complexity did not help here.
- The error bars in the comparison chart overlap, so treat "best algorithm" as a mild lead rather than a clear win.

---

## 4. Quick start

### Option A: Google Colab (easiest, nothing to install)

1. Open the notebook in Colab.
2. Download `Database_register.xlsx` from the GSTRIDE dataset (see [The data](#5-the-data)).
3. Upload it to the Colab session using the folder icon on the left, so it lands at `/content/Database_register.xlsx`.
4. Run all cells from top to bottom.

### Option B: On your own machine

```bash
git clone https://github.com/CollinsLemeke/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/gstride_psychointelligence.ipynb
```

Place the data file at `data/Database_register.xlsx`.

The notebook looks for the file in four places automatically (`data/`, `../data/`, `/content/`, and the current folder), so it works from the repo root, from a subfolder, or from Colab without any edits.

Suggested repository layout:

```
.
├── data/
│   └── Database_register.xlsx      # download separately, not stored in this repo
├── notebooks/
│   └── gstride_psychointelligence.ipynb
├── requirements.txt
└── README.md
```

---

## 5. The data

**GSTRIDE**, a public dataset of 163 older adults. Each person was assessed clinically and then walked while wearing a single inertial sensor on the foot, which recorded spatio-temporal gait parameters such as stride length, stride time, speed, foot angles and clearance. The dataset also includes the standard clinical assessments used here (GDS, Fried phenotype and the TUG test).

Reported characteristics of the cohort: **163 participants, mean age 82.6 years (SD 6.2), roughly 72% women**. The notebook loads 163 rows and 60 columns.

Source paper:

> García-de-Villa, S., Jiménez, A. R., del-Ama, A. J., et al. (2023). *A database with frailty, functional and inertial gait metrics for the research of fall causes in older adults.* **Scientific Data**, 10, 566. https://doi.org/10.1038/s41597-023-02428-0

The data file is **not included in this repository**. Please download it from the source and follow the dataset's own terms of use.

---

## 6. Glossary for beginners

| Term | What it means in plain English |
|---|---|
| **IMU** | Inertial Measurement Unit. A small motion sensor, like the one in a phone or fitness tracker, worn on the foot here. |
| **GDS** | Global Deterioration Scale. A clinical score for stages of cognitive decline. A score of 3 or more is used here as "signs of decline". |
| **Fried phenotype** | A standard five-part checklist for physical frailty. A score of 2 or more is used here as "frail". |
| **TUG** | Timed Up and Go. A simple clinical test: stand up from a chair, walk 3 metres, turn, come back, sit down. Longer time means worse mobility. |
| **Clearance** | How high the foot lifts off the ground during a step. Low clearance means shuffling. |
| **Stride time variability** | How inconsistent the rhythm of walking is from stride to stride. |
| **Feature** | One input measurement the model is allowed to use. |
| **Target** | The yes or no answer the model is trying to predict. |
| **Class imbalance** | When one answer is much rarer than the other. Rarer means fewer examples to learn from, so harder. |
| **Correlation** | How strongly two measurements move together. |
| **VIF** | Variance Inflation Factor, a redundancy meter. Above 5 is a warning, above 10 is severe. |
| **Imputation** | Filling in a missing value. Here, with the column's median (its middle value). |
| **Cross-validation** | Splitting people into groups, training on most and testing on the rest, then rotating so everyone gets a turn as the test set. It gives a fairer score than a single split. |
| **ROC-AUC** | One number summarising how well the model separates the two groups. 0.5 is guessing, 1.0 is perfect. |
| **Confusion matrix** | A 2x2 table of what the model got right and wrong: cases caught, cases missed, false alarms, correct all-clears. |
| **Odds ratio (OR)** | How much the odds of a condition change when a measurement goes up. Above 1 raises the odds, below 1 lowers them. |
| **p-value** | A rough check on whether a result is likely to be real rather than noise. Below 0.05 is the usual bar. |

---

## 7. Walkthrough of the notebook

The notebook runs in 14 steps. Each has a short explanation and at least one chart.

**Step 0. Set up the tools.** Loads the libraries and defines a consistent chart style so every figure looks the same.

**Step 1. Load the data.** Finds the Excel file wherever it lives, then reads the real column names from row 2 and the records from row 4 onwards. Result: 163 people, 60 columns.

**Step 2. First look.** A doughnut chart showing how much of the raw table is actually filled in. Most of it is complete.

**Step 3. Clean the data.** Three small but important fixes:

- The fall label contains both `"NO"` and `"NO "` with a trailing space. Left alone, that single group would be counted as two. Trimmed and converted to 1 and 0.
- Age is stored as bands such as `70-74`. The midpoint (72) is used.
- The three yes or no targets are built from GDS, the fall answer and the Fried score.

**Step 4. Visualise the targets.** How many people have each condition:

| Target | People with the condition | Share |
|---|---|---|
| Dementia | 43 of 163 | 26% |
| Falls | 86 of 163 | 53% |
| Frailty | 49 of 163 | 30% |

Falls is close to an even split. Dementia is the rarest, so its results deserve the most scepticism.

**Step 5. Who is in the study.** Sex, cognitive status and age bands. Mostly women, mostly aged 80 and over. This tells you who the conclusions can reasonably apply to.

**Step 6. Choose features and remove overlaps.** Ten candidate measurements go in. A correlation heatmap and VIF scores reveal that speed, stride length, cadence, clearance variability and swing all carry roughly the same "how well does this person walk" information. Keeping all of them would produce explanations that contradict each other.

The decision: keep two walking representatives (**step speed** and **foot clearance**), drop the duplicates, and keep the independent measures. Final feature set of six:

`age`, `bmi`, `tug`, `step_speed`, `clearance`, `stridetime_std`

After cleaning, every VIF is comfortably under 5 (highest is 2.9 for step speed), which means the redundancy is gone.

**Step 7. Handle missing values.** Only the TUG column has gaps. They are filled with the median **inside the modelling pipeline**, so the fill value is learned only from training data and never leaks from the test set. This is a small detail that many tutorials get wrong.

**Step 8. See the signal with your own eyes.** Box plots comparing the decline group with the healthy group on TUG time and walking speed. The two groups sit at visibly different heights, so there is a real signal to find before any model is built.

**Step 9. How training and testing work.** With only 163 people, one train and test split would be unreliable. The notebook uses 5-fold cross-validation repeated 10 times, illustrated with a simple grid diagram.

**Step 10. Train and compare.** Three algorithms across three tasks, nine runs in total:

| Algorithm | In one line |
|---|---|
| Logistic Regression | Simple and fully explainable |
| Random Forest | A voting committee of decision trees |
| Gradient Boosting | A perfectionist that keeps fixing its own mistakes |

All three sit inside a pipeline that imputes, then scales, then models. Logistic Regression and Random Forest use balanced class weights so the rarer group is not ignored.

**Step 11. A closer look at dementia.** The ROC curve and confusion matrix for the winning model. 31 of 43 cases caught, 12 missed.

**Step 12. What the model learned.** Odds ratios for each of the six features. See the next section.

**Step 13. How the three tasks relate.** Each person gets a risk score for all three tasks, and those scores are compared. This is the most interesting step in the notebook.

**Step 14. Findings and limits.**

---

## 8. What the model learned

Odds ratios from the logistic regression on the dementia task, with all features standardised so they can be compared fairly:

| Feature | Odds ratio | Direction | Statistically reliable (p < 0.05) |
|---|---|---|---|
| TUG time | 2.34 | Raises the odds | **Yes** |
| Age | 1.45 | Raises the odds | No |
| Step speed | 0.88 | Lowers the odds | No |
| Stride time variability | 0.69 | Lowers the odds | No |
| Foot clearance | 0.63 | Lowers the odds | No |
| BMI | 0.52 | Lowers the odds | **Yes** |

**How to read this.** A slower Timed Up and Go is by far the strongest sign of cognitive decline. Faster walking and higher foot clearance point the other way, which matches the hesitant, shuffling pattern a clinician would recognise. That is reassuring: the model found something medically sensible rather than a statistical fluke.

**Two results that need care before anyone quotes them:**

- **Only TUG and BMI clear the p < 0.05 bar.** The other four directions are suggestive, not established. With 163 people and 43 positive cases, that is expected. Please do not report the non-significant odds ratios as findings.
- **Stride time variability points the "wrong" way.** In the wider literature, more variable stride timing is usually associated with worse cognition, not better. Here it appears to lower the odds. It is not statistically reliable, and it may be an artefact of the small sample or of how it interacts with TUG. It is flagged here rather than quietly smoothed over.

---

## 9. The main finding

All three tasks use the **same six measurements**. So do they just flag the same people?

- **Falls and frailty agree almost perfectly** (correlation around 0.9). For practical purposes they are one physical signal wearing two hats.
- **Dementia agrees only moderately** (around 0.5 to 0.7). Related, but clearly its own thing.

That gap is the contribution. If cognitive decline were simply frailty in disguise, its risk score would track frailty almost exactly. It does not. **The walk appears to carry brain-specific information beyond plain physical frailty**, which is the core claim of the Psychointelligence framing.

---

## 10. Honest limitations

Please read this section before citing anything above.

- **This is a screening signal, not a diagnosis.** Only a qualified clinician can diagnose dementia. Nothing here should be used to make a clinical decision about a real person.
- **Small sample.** 163 people, of whom 43 show signs of decline. The findings need replication on a larger and more varied group before they mean much.
- **Narrow cohort.** Mostly women, mostly aged 80 and over, recruited in one study. The results may not transfer to younger, male or more diverse populations.
- **Snapshot, not a prediction.** Walking and cognition were measured at the same time. This work cannot yet show that walking predicts *future* decline.
- **Association, not cause.** A slower TUG is linked to cognitive decline here. That does not mean one causes the other.
- **The feature selection saw all the data.** Correlation and VIF pruning were done on the full dataset before cross-validation. This is a mild form of optimistic bias, because the test folds influenced which features were kept.
- **The odds ratio model was fitted on all 163 people**, so those coefficients describe this dataset rather than a held-out sample.
- **The reported numbers are not nested-CV numbers.** Choosing the best algorithm per task by looking at cross-validation scores will flatter those same scores slightly.

The fix for the last three points is the same, and it is the top item on the roadmap.

---

## 11. Roadmap

Ranked by how much each would improve the credibility of the results:

1. **Nested cross-validation.** Move feature selection and algorithm choice inside the inner loop, so the reported score comes from data the whole procedure has never seen. This is what turns these numbers into publication-grade numbers.
2. **Confidence intervals on every score.** Report AUC with a bootstrap interval rather than a single mean, so readers can judge how much the difference between algorithms means.
3. **Threshold tuning for screening.** A screening tool usually wants a high catch rate and tolerates more false alarms. The 0.5 default is not the right operating point. Pick the threshold from a stated clinical preference and report performance there.
4. **External validation.** Test on a second, independent cohort. This is the single strongest evidence that the signal is real.
5. **Calibration check.** Verify that a predicted risk of 0.7 really does correspond to about a 70% chance, which matters more than AUC for anything used in practice.
6. **Add SHAP explanations.** Per-person explanations alongside the population-level odds ratios, so a clinician can see why this individual was flagged.

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

**Dataset.** Please cite the original GSTRIDE data descriptor:

> García-de-Villa, S., Jiménez, A. R., del-Ama, A. J., et al. (2023). A database with frailty, functional and inertial gait metrics for the research of fall causes in older adults. *Scientific Data*, 10, 566. https://doi.org/10.1038/s41597-023-02428-0

**This analysis.** Collins Lemeke, Centre of Intelligence of Things (CIoTh), University of Greater Manchester. Part of an ongoing line of work on explainable cognitive-decline screening from gait sensor data.

**Contact and links.** GitHub [@CollinsLemeke](https://github.com/CollinsLemeke) · Kaggle [collinslemeke](https://www.kaggle.com/collinslemeke)

---

### A note on intended use

This is a research notebook built for learning and for methodological transparency. It is not a medical device, it has not been clinically validated, and it must not be used to screen, diagnose or advise any real person. If you are worried about your own memory or mobility, or someone else's, please speak to a doctor.
