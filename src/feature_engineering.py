"""Feature engineering: composite scores, grades, and categorical encoding."""

import pandas as pd


def add_composite_scores(df: pd.DataFrame, pass_threshold: int = 40) -> pd.DataFrame:
    """Add engineered features to the student DataFrame.

    New columns:
        - Total Score: sum of math, reading, writing scores
        - Average Score: Total Score / 3
        - math pass, reading pass, writing pass: binary pass/fail per subject
        - Grade: letter grade (A/B/C/D/F) based on Average Score

    Args:
        df: DataFrame with 'math score', 'reading score', 'writing score'.
        pass_threshold: Minimum score to pass a subject (default 40).

    Returns:
        DataFrame with new columns appended.
    """
    df = df.copy()

    # Composite scores
    df["Total Score"] = df["math score"] + df["reading score"] + df["writing score"]
    df["Average Score"] = df["Total Score"] / 3

    # Per-subject pass/fail
    for subject in ["math", "reading", "writing"]:
        df[f"{subject} pass"] = (df[f"{subject} score"] >= pass_threshold).astype(int)

    # Overall pass: pass all three subjects
    df["Overall Pass"] = (df["math pass"] & df["reading pass"] & df["writing pass"]).astype(int)

    # Letter grade based on Average Score
    df["Grade"] = pd.cut(
        df["Average Score"],
        bins=[-1, 60, 70, 80, 90, 101],
        labels=["F", "D", "C", "B", "A"],
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables for ML pipelines.

    Binary encoding for two-class variables, one-hot for multi-class.

    Args:
        df: DataFrame with original categorical columns.

    Returns:
        DataFrame with encoded features (original categoricals dropped).
    """
    df = df.copy()

    # Binary encoding
    df["test_prep_completed"] = (df["test preparation course"] == "completed").astype(int)
    df["lunch_standard"] = (df["lunch"] == "standard").astype(int)
    df["is_female"] = (df["gender"] == "female").astype(int)

    # One-hot encoding for multi-class
    df = pd.get_dummies(df, columns=["race/ethnicity"], prefix="race", drop_first=True)
    df = pd.get_dummies(df, columns=["parental level of education"], prefix="parent_edu", drop_first=True)

    # Drop original categorical columns (now encoded)
    cols_to_drop = ["gender", "lunch", "test preparation course"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df
