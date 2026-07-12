# House Price Prediction (Ames / Kaggle "House Prices")

Regression on the Ames Housing dataset — predicting `SalePrice` from structural and location
features. The competition metric is **RMSE on log(SalePrice)** (i.e. RMSLE), so the target is
log-transformed throughout.

The project has two parts:

- **Transparent baseline** — [`src/baseline.py`](src/baseline.py): a Linear Regression / Ridge
  pipeline (median imputation, standardization, one-hot encoding) with honest hold-out and 5-fold
  cross-validated RMSLE. It loads the data from **OpenML**, so it runs with no Kaggle credentials.
- **AutoGluon pipeline** — [`HousePricePrediction.ipynb`](HousePricePrediction.ipynb): data
  cleaning, feature engineering (age, total-area, polynomial, log-transformed skewed features),
  and an AutoGluon `TabularPredictor` (`best_quality`) ensemble, plus EDA and Kaggle submission
  generation. This notebook expects the Kaggle competition data (downloaded via the Kaggle API).

## Results

Transparent baseline (reproducible via `python src/baseline.py`, seed = 42):

| Model | 5-fold CV RMSLE | Hold-out RMSLE |
|-------|:---:|:---:|
| Linear Regression | 0.153 ± 0.047 | 0.128 |
| Ridge (α = 10) | **0.147 ± 0.039** | 0.136 |

The AutoGluon ensemble in the notebook is expected to improve on this baseline; its per-model
leaderboard (validation RMSLE) is printed by the notebook's modeling section. Reporting the linear
baseline alongside it shows how much the added model complexity is actually worth.

## Repository layout

```
├── HousePricePrediction.ipynb   # full pipeline: cleaning, feature engineering, AutoGluon, EDA
├── src/baseline.py              # reproducible Linear/Ridge baseline with reported RMSLE
├── requirements.txt
└── README.md
```

## Getting started

```bash
pip install -r requirements.txt

# Transparent baseline — no credentials needed (data pulled from OpenML)
python src/baseline.py

# Full AutoGluon pipeline — open the notebook; it downloads the Kaggle competition
# data via the Kaggle API (set up ~/.kaggle/kaggle.json first) and trains AutoGluon.
```

## Notes

- **Metric:** RMSLE = RMSE on `log1p(SalePrice)`. Lower is better.
- The baseline uses the Ames data from OpenML (`data_id=42165`); the notebook uses the Kaggle
  competition train/test split for leaderboard submission.
