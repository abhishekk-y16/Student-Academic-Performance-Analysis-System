"""Data ingestion, validation, and summary statistics."""

import os
import pandas as pd


def load_data(path: str = None) -> pd.DataFrame:
    """Load the student performance dataset from CSV.

    Args:
        path: Path to the CSV file. Defaults to data/StudentsPerformance.csv
              relative to the project root.

    Returns:
        Raw DataFrame with original columns and dtypes.
    """
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "data", "StudentsPerformance.csv")
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run data quality checks and clean the DataFrame.

    Checks performed:
        1. Null value counts per column
        2. Duplicate row counts
        3. Dtype verification (score columns must be numeric)

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Cleaned DataFrame with duplicates removed (if any).
    """
    print("=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)

    # Shape
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Null check
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\nNull values per column (total: {total_nulls}):")
    print(null_counts.to_string())

    # Duplicate check
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  → Removed {dup_count} duplicates. New shape: {df.shape}")

    # Dtype verification
    score_cols = ["math score", "reading score", "writing score"]
    print("\nData types:")
    for col in df.columns:
        dtype = df[col].dtype
        status = "✓" if col not in score_cols else ("✓ numeric" if pd.api.types.is_numeric_dtype(df[col]) else "✗ NOT NUMERIC")
        print(f"  {col}: {dtype} {status}")

    # Convert score columns to numeric if needed
    for col in score_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            print(f"  → Converted {col} to numeric")

    print("\n" + "=" * 50)
    return df


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns.

    Args:
        df: DataFrame (raw or engineered).

    Returns:
        DataFrame with count, mean, std, min, 25%, 50%, 75%, max.
    """
    return df.describe()
