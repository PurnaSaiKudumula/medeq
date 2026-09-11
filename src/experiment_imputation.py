"""
experiment_imputation.py

PURPOSE: A senior asked whether using smarter imputation (filling missing
values with a meaningful estimate) beats our current approach (filling
with 0 + a missing-indicator flag). Rather than guessing, this script
actually tests three strategies on the same data/model/train-test split
and reports real accuracy/precision/recall/F1 for each — so the
comparison is honest, not assumed.

STRATEGY A (current/baseline): fill missing telemetry with 0, add a
    "*_was_missing" flag column so the model knows it was unknown.
STRATEGY B (global mean imputation): fill missing telemetry with the
    OVERALL average value of that field across all devices, still with
    the missing flag kept.
STRATEGY C (category-wise mean imputation): fill missing telemetry with
    the average value of that field WITHIN THE SAME DEVICE CATEGORY
    (e.g. missing vibration on a Cardiovascular device gets the average
    vibration seen on OTHER Cardiovascular devices, not the global
    average across all device types) — a more informed guess, since
    different device categories genuinely behave differently.

NOTE: in our synthetic dataset, "missing" telemetry fields are ones a
device category doesn't track AT ALL (e.g. Orthopedic Devices have no
temperature field defined in device_schema.json) — so for those fields,
every row for that category is missing, which means a "category-wise
mean" would have nothing real to average either. This experiment is
still worth running because it tells us plainly whether that's true,
rather than assuming it.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb


TELEMETRY_FIELDS = ["temperature", "vibration", "voltage", "hours_used"]


def build_features_strategy(df: pd.DataFrame, strategy: str):
    """
    Builds the feature matrix using one of three missing-value strategies.
    Everything else (category encoding, missing flags) stays identical
    across strategies so the comparison is fair — the ONLY thing that
    changes is what value fills the gap.
    """
    df = df.copy()

    for field in TELEMETRY_FIELDS:
        if field not in df.columns:
            df[field] = np.nan

        df[f"{field}_was_missing"] = df[field].isna().astype(int)

        if strategy == "zero_fill":
            df[field] = df[field].fillna(0)

        elif strategy == "global_mean":
            global_mean = df[field].mean()
            # If EVERY value is missing, global_mean itself is NaN —
            # fall back to 0 in that edge case so training doesn't break.
            fill_value = global_mean if not pd.isna(global_mean) else 0
            df[field] = df[field].fillna(fill_value)

        elif strategy == "category_mean":
            category_means = df.groupby("classification")[field].transform("mean")
            df[field] = df[field].fillna(category_means)
            # If a whole category has zero real values for this field,
            # category_means will still be NaN for those rows — fall
            # back to 0 for any leftover gaps.
            df[field] = df[field].fillna(0)

    encoder = LabelEncoder()
    df["classification_encoded"] = encoder.fit_transform(df["classification"])

    feature_columns = (
        TELEMETRY_FIELDS
        + [f"{f}_was_missing" for f in TELEMETRY_FIELDS]
        + ["age_fraction", "classification_encoded"]
    )
    X = df[feature_columns]
    y = df["failed"]
    return X, y


def run_experiment(df: pd.DataFrame, strategy: str):
    """
    Trains a classifier under one strategy and returns its metrics.
    Same train/test split (random_state=42) is used every time, so
    differences in results come from the imputation strategy alone,
    not from a lucky/unlucky data split.
    """
    X, y = build_features_strategy(df, strategy)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "strategy": strategy,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
    }


def main():
    """
    Entry point for the imputation experiment.

    Steps:
      1. Loads the base synthetic telemetry CSV.
      2. Prints how much real (non-NaN) data exists per telemetry field —
         an upfront sanity check on whether category-mean imputation even
         has anything real to average.
      3. Runs run_experiment() once per strategy (zero_fill, global_mean,
         category_mean) using the same train/test split so the only
         variable is the imputation method.
      4. Prints a side-by-side comparison and the best strategy by F1.

    Run with:  python src/experiment_imputation.py
    """
    df = pd.read_csv("data/synthetic_telemetry.csv")

    # First, print how much real (non-NaN) data actually exists per
    # field — this tells us upfront whether category-wise mean
    # imputation even has anything real to average.
    print("=== Real data availability per telemetry field ===")
    for field in TELEMETRY_FIELDS:
        if field in df.columns:
            pct_present = df[field].notna().mean() * 100
            print(f"{field}: {pct_present:.1f}% present")
    print()

    results = []
    for strategy in ["zero_fill", "global_mean", "category_mean"]:
        result = run_experiment(df, strategy)
        results.append(result)
        print(f"=== {strategy} ===")
        for k, v in result.items():
            if k != "strategy":
                print(f"  {k}: {v}")
        print()

    results_df = pd.DataFrame(results)
    print("=== Side-by-side comparison ===")
    print(results_df.to_string(index=False))

    best = results_df.loc[results_df["f1"].idxmax()]
    print(f"\nBest strategy by F1 score: {best['strategy']} (F1={best['f1']})")


if __name__ == "__main__":
    main()
