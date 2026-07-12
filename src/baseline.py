"""Reproducible linear baseline for the Ames house-price regression.

The notebook trains an AutoGluon ensemble; this script provides the transparent baseline it
should be measured against. It loads the Ames Housing data from OpenML (the same dataset as the
Kaggle "House Prices" competition), log-transforms the target, and reports honest hold-out and
5-fold cross-validated RMSLE for Linear Regression and Ridge — the competition metric is RMSE on
log(SalePrice), i.e. RMSLE.

Run:  python src/baseline.py
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def load_ames():
    """Ames Housing (Kaggle House Prices training set) from OpenML, id 42165."""
    frame = fetch_openml(data_id=42165, as_frame=True).frame
    y = np.log1p(frame["SalePrice"].astype(float))
    X = frame.drop(columns=[c for c in ("SalePrice", "Id") if c in frame.columns])
    return X, y


def build_pipeline(model) -> Pipeline:
    """Median/most-frequent imputation, scaling for numerics, one-hot for categoricals."""
    from sklearn.compose import make_column_selector as selector

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
             selector(dtype_include="number")),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]),
             selector(dtype_exclude="number")),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    X, y = load_ames()
    print(f"Ames data: {X.shape[0]} rows x {X.shape[1]} features (target = log1p(SalePrice))")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"{'model':<20}{'5-fold CV RMSLE':>22}{'hold-out RMSLE':>18}")
    print("-" * 60)
    for name, model in [("LinearRegression", LinearRegression()), ("Ridge(alpha=10)", Ridge(alpha=10.0))]:
        pipe = build_pipeline(model)
        neg = cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=kf)
        cv_mean, cv_std = -neg.mean(), neg.std()
        pipe.fit(X_tr, y_tr)
        holdout = float(np.sqrt(mean_squared_error(y_te, pipe.predict(X_te))))
        print(f"{name:<20}{cv_mean:>10.4f} ± {cv_std:<7.4f}{holdout:>18.4f}")


if __name__ == "__main__":
    main()
