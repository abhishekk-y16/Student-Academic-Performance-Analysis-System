"""Machine learning models for student performance prediction."""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from src.feature_engineering import encode_categoricals

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Feature columns after encoding (used for prediction consistency)
FEATURE_COLS = None  # Set during training


def _get_feature_matrix(df: pd.DataFrame):
    """Prepare encoded feature matrix (X) from a DataFrame.

    Encodes categoricals and drops target/non-feature columns.

    Returns:
        (X DataFrame, list of feature column names)
    """
    encoded = encode_categoricals(df)
    drop_cols = [
        "math score", "reading score", "writing score",
        "Total Score", "Average Score", "Grade",
        "math pass", "reading pass", "writing pass", "Overall Pass",
    ]
    feature_cols = [c for c in encoded.columns if c not in drop_cols]
    return encoded[feature_cols], feature_cols


def train_classifier(df: pd.DataFrame, target: str = "Overall Pass"):
    """Train classification models to predict pass/fail.

    Models: Logistic Regression, Random Forest Classifier.

    Args:
        df: DataFrame with composite scores (from add_composite_scores).
        target: Binary target column name.

    Returns:
        dict with model objects, evaluation metrics, and feature importance.
    """
    X, feature_cols = _get_feature_matrix(df)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["logistic_regression"] = {
        "model": lr,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr, zero_division=0),
        "recall": recall_score(y_test, y_pred_lr, zero_division=0),
        "f1": f1_score(y_test, y_pred_lr, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr),
    }

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["random_forest_classifier"] = {
        "model": rf,
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "recall": recall_score(y_test, y_pred_rf, zero_division=0),
        "f1": f1_score(y_test, y_pred_rf, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf),
        "feature_importance": dict(zip(feature_cols, rf.feature_importances_)),
    }

    results["feature_cols"] = feature_cols
    return results


def train_regressor(df: pd.DataFrame):
    """Train regression models to predict individual subject scores.

    Models: Linear Regression, Random Forest Regressor (per subject).

    Args:
        df: DataFrame with composite scores.

    Returns:
        dict with model objects and evaluation metrics per subject.
    """
    X, feature_cols = _get_feature_matrix(df)
    targets = ["math score", "reading score", "writing score"]
    results = {}

    for target in targets:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        results[target] = {
            "linear_regression": {
                "model": lr,
                "r2": r2_score(y_test, y_pred_lr),
                "mae": mean_absolute_error(y_test, y_pred_lr),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            },
            "random_forest": {
                "model": rf,
                "r2": r2_score(y_test, y_pred_rf),
                "mae": mean_absolute_error(y_test, y_pred_rf),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                "feature_importance": dict(zip(feature_cols, rf.feature_importances_)),
            },
        }

    results["feature_cols"] = feature_cols
    return results


def train_and_save_models(df: pd.DataFrame, model_dir: str = None):
    """Train all models and serialize to disk.

    Args:
        df: DataFrame with composite scores (from add_composite_scores).
        model_dir: Output directory for .pkl files. Defaults to models/.

    Returns:
        dict with classifier and regressor results.
    """
    if model_dir is None:
        model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    print("Training classification models...")
    clf_results = train_classifier(df)
    print("Training regression models...")
    reg_results = train_regressor(df)

    # Save best models (Random Forest for both tasks)
    joblib.dump(clf_results["random_forest_classifier"]["model"],
                os.path.join(model_dir, "rf_classifier.pkl"))
    joblib.dump(clf_results["feature_cols"],
                os.path.join(model_dir, "feature_cols.pkl"))

    for target in ["math score", "reading score", "writing score"]:
        safe_name = target.replace(" ", "_")
        joblib.dump(reg_results[target]["random_forest"]["model"],
                    os.path.join(model_dir, f"rf_regressor_{safe_name}.pkl"))

    # Print evaluation summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS (Pass/Fail)")
    print("=" * 60)
    for name in ["logistic_regression", "random_forest_classifier"]:
        r = clf_results[name]
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Accuracy:  {r['accuracy']:.3f}")
        print(f"  Precision: {r['precision']:.3f}")
        print(f"  Recall:    {r['recall']:.3f}")
        print(f"  F1 Score:  {r['f1']:.3f}")

    print("\n" + "=" * 60)
    print("REGRESSION RESULTS (Score Prediction)")
    print("=" * 60)
    for target in ["math score", "reading score", "writing score"]:
        print(f"\n{target.title()}:")
        for name in ["linear_regression", "random_forest"]:
            r = reg_results[target][name]
            print(f"  {name.replace('_', ' ').title()}:")
            print(f"    R²:   {r['r2']:.3f}")
            print(f"    MAE:  {r['mae']:.2f}")
            print(f"    RMSE: {r['rmse']:.2f}")

    print(f"\nModels saved to: {model_dir}")
    return {"classifier": clf_results, "regressor": reg_results}


def load_models(model_dir: str = None):
    """Load serialized models from disk.

    Returns:
        dict with classifier model, regressor models, and feature columns.
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    models = {
        "classifier": joblib.load(os.path.join(model_dir, "rf_classifier.pkl")),
        "feature_cols": joblib.load(os.path.join(model_dir, "feature_cols.pkl")),
    }
    for target in ["math_score", "reading_score", "writing_score"]:
        models[target] = joblib.load(os.path.join(model_dir, f"rf_regressor_{target}.pkl"))

    return models


def predict_scores(student_profile: dict, models: dict = None):
    """Predict scores for a single student profile.

    Args:
        student_profile: dict with keys: gender, race/ethnicity,
            parental level of education, lunch, test preparation course.
        models: Pre-loaded models dict. If None, loads from disk.

    Returns:
        dict with predicted math, reading, writing scores, average, and pass/fail.
    """
    if models is None:
        models = load_models()

    # Create single-row DataFrame from profile
    profile_df = pd.DataFrame([student_profile])

    # Encode using the same pipeline
    encoded = encode_categoricals(profile_df)

    # Ensure all feature columns exist (fill missing one-hot columns with 0)
    feature_cols = models["feature_cols"]
    for col in feature_cols:
        if col not in encoded.columns:
            encoded[col] = 0
    X = encoded[feature_cols]

    # Predict scores
    math_pred = float(np.clip(models["math_score"].predict(X)[0], 0, 100))
    reading_pred = float(np.clip(models["reading_score"].predict(X)[0], 0, 100))
    writing_pred = float(np.clip(models["writing_score"].predict(X)[0], 0, 100))

    avg_pred = (math_pred + reading_pred + writing_pred) / 3

    # Predict pass/fail
    pass_pred = int(models["classifier"].predict(X)[0])

    return {
        "math score": round(math_pred, 1),
        "reading score": round(reading_pred, 1),
        "writing score": round(writing_pred, 1),
        "Average Score": round(avg_pred, 1),
        "Overall Pass": "Pass" if pass_pred == 1 else "Fail",
    }
