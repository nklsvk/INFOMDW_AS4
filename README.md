# Reweighing: fairness evaluation on Adult, German Credit and COMPAS

This folder evaluates **Reweighing** (Kamiran and Calders, 2012, *Data preprocessing techniques
for classification without discrimination*, Knowledge and Information Systems 33(1), 1-33),
using the implementation in this repository (`aif360.sklearn.preprocessing.Reweighing`).

The notebook `reweighing_fairness.ipynb` has already been run, and its outputs are saved,
so the results can be read on GitHub without running anything.

## What it does

1. Computes DP_Diff, DIR and consistency on the original labels of each dataset (no model).
2. Makes a 70/30 train/test split, stratified on protected group and label together.
3. Computes the Reweighing weights on the training part only.
4. Trains logistic regression with and without the weights. The protected attribute is not a feature.
5. Reports DP_Diff, DIR, consistency, accuracy and balanced accuracy on the test part.
6. Repeats steps 2 to 5 for 10 random splits (seeds 0 to 9) and reports mean and standard deviation.
7. Extra check: the same experiment with the protected attribute used as a feature.

| Dataset | Source | Protected attribute (privileged) | Favourable label |
|---|---|---|---|
| Adult | UCI, via github.com/jbrownlee/Datasets | sex (male) | income > 50K |
| German Credit | UCI, via github.com/jbrownlee/Datasets | age (25 or older) | good credit |
| COMPAS | github.com/propublica/compas-analysis | race (Caucasian) | no re-offence within 2 years |

All data are downloaded automatically by the notebook. Nothing has to be downloaded by hand.

## How to reproduce

### Option A: Google Colab (no installation)

1. Open https://colab.research.google.com and choose **File > Open notebook > GitHub**.
2. Paste the link to `reweighing_fairness.ipynb` in this folder.
3. Click **Runtime > Run all**. The first cell installs AIF360. The run takes about one minute.

### Option B: on your own computer

```bash
git clone https://github.com/nklsvk/INFOMDW_AS4.git
cd AIF360/reweighing_reproduction
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace reweighing_fairness.ipynb
```

The last command runs the whole notebook and saves the outputs inside it. You can also open it
with `jupyter notebook` and choose **Run > Run All Cells**.

## Output

The notebook prints all tables and writes them to `results/`:

- `before.csv`: measures on the original labels (report Table 2)
- `results_summary.csv`: mean and standard deviation per dataset and method (report Table 3)
- `results_per_seed.csv`: every single run

## Parameters

| Setting | Value |
|---|---|
| Reweighing | no parameters apart from the protected attribute |
| Train/test split | 70/30, stratified on group and label, seeds 0 to 9 |
| Encoding | one-hot for categorical columns, StandardScaler, both fitted on the training part only |
| Classifier | `LogisticRegression(max_iter=2000)`, all other settings default (lbfgs, C = 1.0) |
| Decision threshold | 0.5 |
| Consistency | k = 5 nearest neighbours on the standardised features |
| Adult | rows with missing values removed, `fnlwgt` dropped |
| COMPAS | standard ProPublica filtering (screening within 30 days, charge degree not "O") |

Tested with Python 3.13, aif360 0.6.1, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3.
