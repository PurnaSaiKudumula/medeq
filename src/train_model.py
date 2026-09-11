"""
train_model.py

PURPOSE OF THIS FILE:
Takes the synthetic telemetry dataset (built by synthetic_telemetry.py)
and trains TWO models on it:

  1. A CLASSIFIER (XGBoost) that predicts "failed" (0 or 1) — this answers
     the brief's core requirement: "identifies equipment failure risk."

  2. A REGRESSOR (XGBoost) that predicts "true_failure_probability" as a
     continuous number — this is the basis for our Remaining Useful Life
     (RUL) feature: turning a raw probability into an estimated "days
     until failure" countdown, instead of just a yes/no flag.

It also runs TreeSHAP on the classifier so we can explain WHY each
prediction was made — this is the "explains key drivers" requirement
from the brief, and it's what feeds the SHAP-to-SOP dispatch feature.

Finally, it reports accuracy/precision/recall/F1 — required by the
official evaluation rubric ("Model Performance and Evaluation").
"""

import pandas as pd
import numpy as np
import json
import os
import joblib  # used to save trained models to disk so the API can load them later

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the synthetic telemetry CSV produced by synthetic_telemetry.py.
    Kept as its own tiny function so if the data source ever changes
    (e.g. a real hospital's telemetry export instead of synthetic data),
    only this function needs to change — nothing downstream does.
    """
    return pd.read_csv(csv_path)


def build_features(df: pd.DataFrame):
    """
    Turns the raw dataframe into a clean (X, y_classification, y_regression)
    set ready for model training.

    WHY WE ENCODE 'classification' AS A NUMBER:
    ML models like XGBoost need numeric input — they can't directly use a
    text category like "Cardiovascular Devices". LabelEncoder converts each
    unique category into a distinct integer (e.g. 0, 1, 2...) so the model
    can use device category as a real, usable feature without us manually
    assigning numbers ourselves.

    WHY WE FILL MISSING SENSOR COLUMNS WITH 0 + A "WAS_MISSING" FLAG:
    Not every device category has every sensor field (see device_schema.json
    — e.g. Orthopedic Devices don't track temperature). Rather than leaving
    these as NaN (which breaks many models) or silently guessing a fake
    value, we fill them with 0 AND add a separate "*_was_missing" column.
    This is the "missing-indicator flag" approach discussed earlier — the
    model can explicitly learn "temperature wasn't tracked for this device
    type" as its own signal, rather than being confused by a fake zero
    that looks like a real low reading.
    """
    df = df.copy()

    # All possible telemetry fields across every device category.
    telemetry_fields = ["temperature", "vibration", "voltage", "hours_used"]

    for field in telemetry_fields:
        if field not in df.columns:
            df[field] = np.nan
        # missing-indicator flag: 1 if this field wasn't recorded for this
        # device's category, 0 if it was.
        df[f"{field}_was_missing"] = df[field].isna().astype(int)
        # Now safe to fill the actual missing values with 0 — the flag
        # column above preserves the "this was unknown" information.
        df[field] = df[field].fillna(0)

    # Encode device category as a number the model can use.
    category_encoder = LabelEncoder()
    df["classification_encoded"] = category_encoder.fit_transform(df["classification"])

    # Encode manufacturer_risk_tier — a REAL feature derived from actual
    # events.csv + manufacturers.csv recall history (see
    # real_data_features.py), not synthetic. This is what makes
    # manufacturers.csv and events.csv genuinely USED in the final
    # prediction, not just analyzed and set aside.
    if "manufacturer_risk_tier" not in df.columns:
        df["manufacturer_risk_tier"] = "low"
    tier_map = {"low": 0, "medium": 1, "high": 2}
    df["manufacturer_risk_tier_encoded"] = df["manufacturer_risk_tier"].map(tier_map).fillna(0)

    feature_columns = (
        telemetry_fields
        + [f"{f}_was_missing" for f in telemetry_fields]
        + ["age_fraction", "classification_encoded", "manufacturer_risk_tier_encoded"]
    )

    X = df[feature_columns]
    y_classification = df["failed"]
    y_regression = df["true_failure_probability"]

    return X, y_classification, y_regression, category_encoder, feature_columns


def tune_classifier(X_train, y_train):
    """
    HYPERPARAMETER TUNING for the classifier using RandomizedSearchCV.

    WHAT THIS ACTUALLY DOES: instead of us guessing fixed values for
    n_estimators, max_depth, learning_rate etc. (like train_classifier()
    does), this tries many different RANDOM COMBINATIONS of these values,
    trains a model for each combination using cross-validation (splitting
    the training data into folds and testing on each), and keeps whichever
    combination scored best on F1 (not accuracy — see why below).

    WHY RandomizedSearchCV INSTEAD OF GridSearchCV:
    GridSearchCV tries EVERY possible combination of every parameter —
    with 5+ parameters and several values each, that's thousands of
    combinations, far too slow for a hackathon timeline. RandomizedSearchCV
    samples a fixed number (n_iter) of random combinations instead —
    much faster, and in practice finds results nearly as good as a full
    grid search for tree-based models like XGBoost.

    WHY WE SCORE ON F1, NOT ACCURACY:
    We already learned the hard way (scale_pos_weight experiment) that
    optimizing for raw accuracy can quietly produce a model that just
    predicts the majority class. Scoring the search on F1 keeps the
    tuning honest — it won't reward a combination that looks good only
    because it ignores real failures.

    WHY 5-FOLD CROSS-VALIDATION (cv=5):
    Instead of one single train/validation split (which could be a lucky
    or unlucky split), the training data is divided into 5 chunks; the
    model trains on 4 and validates on the 5th, five times, rotating
    which chunk is held out. The average score across all 5 rounds is
    more reliable than any single split — this is standard practice for
    trustworthy tuning, not just a one-off experiment.
    """
    param_distributions = {
        "n_estimators": [100, 150, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }

    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    base_model = xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=40,          # try 40 random combinations — good coverage
                             # without taking too long for a hackathon
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,           # use all available CPU cores to speed this up
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best hyperparameters found: {search.best_params_}")
    print(f"Best cross-validated F1 during search: {search.best_score_:.4f}")

    return search.best_estimator_


def train_classifier(X_train, y_train):
    """
    Trains the XGBoost classifier that predicts failure risk (0/1).

    WHY THESE SPECIFIC HYPERPARAMETERS:
    - n_estimators=150: number of small trees to build in sequence.
      150 is a reasonable middle ground — enough trees to learn real
      patterns without taking long to train or overfitting on a
      20,000-row dataset.
    - max_depth=4: keeps each individual tree shallow/simple. Shallow
      trees generalize better and are also easier for SHAP to explain
      clearly — deep trees create more complex, harder-to-summarize
      interactions.
    - learning_rate=0.1: controls how much each new tree corrects the
      previous ones. 0.1 is a standard, safe default — small enough to
      avoid wild overcorrection, large enough to converge in a
      reasonable number of trees.
    - eval_metric="logloss": the standard scoring function for binary
      classification with probability outputs, which is what we need
      for risk percentages.
    - scale_pos_weight: THIS IS THE CLASS-IMBALANCE FIX. Our data has
      roughly 30% "failed" and 70% "not failed" — without correction,
      XGBoost naturally leans toward predicting the majority class
      ("not failed") because that alone gets it a high accuracy score
      cheaply. scale_pos_weight tells the model "treat each failure
      example as worth (majority_count / minority_count) times more"
      when computing its training error — this forces it to actually
      try harder to catch real failures instead of ignoring them.
      In a hospital safety context, missing a real failure (a false
      negative) is far more costly than a false alarm, so we WANT the
      model biased toward catching failures, not toward raw accuracy.
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_regressor(X_train, y_train):
    """
    Trains the XGBoost regressor used for the RUL (Remaining Useful Life)
    feature. Instead of predicting "will it fail: yes/no," this predicts
    a continuous failure-probability number, which the RUL module (a
    separate file) converts into an estimated "days until failure."

    Same hyperparameter reasoning as the classifier above, just using
    XGBRegressor instead of XGBClassifier since the target is continuous.
    """
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test):
    """
    Computes and prints the metrics explicitly required by the official
    evaluation rubric: accuracy, precision, recall, F1.

    WHY WE REPORT ALL FOUR, NOT JUST ACCURACY:
    Accuracy alone can be misleading, especially with imbalanced classes
    (e.g. if only 10% of devices fail, a model that always predicts
    "won't fail" would show 90% accuracy while being completely useless).
    Precision tells you "of the devices we flagged as high risk, how many
    really were." Recall tells you "of the devices that really were at
    risk, how many did we catch." F1 balances the two. Reporting all four
    honestly reflects real model quality, not just a flattering number.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print("=== Classifier Evaluation ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    print()
    print(classification_report(y_test, y_pred))

    return metrics


def evaluate_regressor(model, X_test, y_test):
    """
    Reports Mean Absolute Error (MAE) for the RUL regression model —
    on average, how far off is our predicted failure-probability number
    from the true value. Lower is better. This is the standard metric
    for regression tasks, parallel to accuracy/F1 for classification.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"=== Regressor Evaluation ===\nMAE: {mae:.3f}\n")
    return {"mae": mae}


def compute_shap_values(model, X_sample):
    """
    Runs TreeSHAP on the trained classifier to get per-prediction feature
    importance — i.e., for each individual device, exactly how much each
    feature (vibration, age, etc.) pushed its risk score up or down.

    WHY TreeExplainer SPECIFICALLY (not the generic SHAP Explainer):
    TreeExplainer is a version of SHAP optimized specifically for
    tree-based models like XGBoost — it's much faster and mathematically
    exact for this model type, versus the generic explainer which is
    slower and only approximate. Since our whole pipeline is built on
    XGBoost, TreeExplainer is the correct, efficient choice.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def main():
    """
    Runs the full training pipeline TWICE — once per model variant — and
    saves both, so the dashboard can let a judge compare them side by
    side rather than us silently picking one number to show.

    WHY TWO VARIANTS, TRAINED AND SAVED SEPARATELY:
    Earlier testing showed a direct tradeoff: a "realistic" synthetic
    label (15% random noise blended in) gives ~95% accuracy with a more
    defensible story ("some real-world unpredictability built in"), while
    a "optimized" fully deterministic label gives ~99% accuracy but is
    easier to fairly criticize as "the label is a direct function of the
    same features the model sees." Rather than hide this tradeoff, we
    keep BOTH models available — this lets us openly show a judge the
    difference and explain the reasoning, which is a stronger, more
    honest position than presenting only one number.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    variants = {
        "realistic": os.path.join(base_dir, "data", "synthetic_telemetry_realistic.csv"),
        "optimized": os.path.join(base_dir, "data", "synthetic_telemetry_optimized.csv"),
    }

    all_metrics = {}

    for variant_name, data_path in variants.items():
        print(f"\n{'='*20} TRAINING VARIANT: {variant_name} {'='*20}")

        df = load_data(data_path)
        X, y_class, y_reg, encoder, feature_columns = build_features(df)

        X_train, X_test, y_class_train, y_class_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        _, _, y_reg_train, y_reg_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )

        print("Training classifier (tuning hyperparameters via RandomizedSearchCV)...")
        clf = tune_classifier(X_train, y_class_train)
        clf_metrics = evaluate_classifier(clf, X_test, y_class_test)

        print("Training RUL regressor...")
        reg = train_regressor(X_train, y_reg_train)
        reg_metrics = evaluate_regressor(reg, X_test, y_reg_test)

        # Save this variant's models with a variant-specific filename so
        # both sets coexist on disk without overwriting each other.
        joblib.dump(clf, os.path.join(models_dir, f"classifier_{variant_name}.pkl"))
        joblib.dump(reg, os.path.join(models_dir, f"regressor_{variant_name}.pkl"))
        joblib.dump(encoder, os.path.join(models_dir, f"category_encoder_{variant_name}.pkl"))
        joblib.dump(feature_columns, os.path.join(models_dir, f"feature_columns_{variant_name}.pkl"))

        all_metrics[variant_name] = {"classifier": clf_metrics, "regressor": reg_metrics}

    # Save a single combined metrics file so the dashboard can show both
    # variants' numbers side by side without recomputing anything.
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nBoth variants trained and saved to {models_dir}")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
