"""Reusable EDA plotting functions for notebooks and dashboard."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Consistent styling
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

PARENTAL_EDUCATION_ORDER = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]


def plot_score_distribution(df: pd.DataFrame, subject: str, ax=None):
    """Histogram with KDE overlay for a single subject score.

    Args:
        df: DataFrame containing the score column.
        subject: Column name (e.g. 'math score').
        ax: Optional matplotlib Axes to draw on.

    Returns:
        matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    sns.histplot(df[subject], bins=20, kde=True, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Distribution of {subject.title()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    # Add mean and median lines
    mean_val = df[subject].mean()
    median_val = df[subject].median()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color="green", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.1f}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    return fig, ax


def plot_subject_comparison(df: pd.DataFrame, ax=None):
    """Side-by-side box plots comparing math, reading, and writing scores.

    Args:
        df: DataFrame with score columns.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    score_data = df[["math score", "reading score", "writing score"]].melt(var_name="Subject", value_name="Score")
    sns.boxplot(data=score_data, x="Subject", y="Score", ax=ax, palette="Set2")
    ax.set_title("Subject-Wise Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=12)

    plt.tight_layout()
    return fig, ax


def plot_gender_comparison(df: pd.DataFrame, subject: str, ax=None):
    """Box plot of a subject score split by gender.

    Args:
        df: DataFrame with 'gender' and score columns.
        subject: Score column name.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    sns.boxplot(data=df, x="gender", y=subject, ax=ax, palette="Set1")
    ax.set_title(f"{subject.title()} by Gender", fontsize=14, fontweight="bold")
    ax.set_xlabel("Gender", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)

    plt.tight_layout()
    return fig, ax


def plot_test_prep_impact(df: pd.DataFrame, ax=None):
    """Box plot of Average Score by test preparation course status.

    Args:
        df: DataFrame with 'test preparation course' and 'Average Score'.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    sns.boxplot(data=df, x="test preparation course", y="Average Score", ax=ax, palette="Set2")
    ax.set_title("Impact of Test Preparation Course on Average Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Test Preparation Course", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)

    plt.tight_layout()
    return fig, ax


def plot_parental_education_impact(df: pd.DataFrame, ax=None):
    """Box plot of Average Score by parental level of education (ordered).

    Args:
        df: DataFrame with 'parental level of education' and 'Average Score'.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    sns.boxplot(
        data=df,
        x="parental level of education",
        y="Average Score",
        order=PARENTAL_EDUCATION_ORDER,
        ax=ax,
        palette="YlOrRd",
    )
    ax.set_title("Impact of Parental Education on Average Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Parental Level of Education", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    return fig, ax


def plot_lunch_impact(df: pd.DataFrame, ax=None):
    """Box plot of Average Score by lunch type (SES proxy).

    Args:
        df: DataFrame with 'lunch' and 'Average Score'.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    sns.boxplot(data=df, x="lunch", y="Average Score", ax=ax, palette="coolwarm")
    ax.set_title("Impact of Lunch Type (SES Proxy) on Average Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lunch Type", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)

    plt.tight_layout()
    return fig, ax


def plot_race_ethnicity_comparison(df: pd.DataFrame, ax=None):
    """Box plot of Average Score by race/ethnicity group.

    Args:
        df: DataFrame with 'race/ethnicity' and 'Average Score'.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    group_order = sorted(df["race/ethnicity"].unique())
    sns.boxplot(data=df, x="race/ethnicity", y="Average Score", order=group_order, ax=ax, palette="viridis")
    ax.set_title("Average Score by Race/Ethnicity Group", fontsize=14, fontweight="bold")
    ax.set_xlabel("Race/Ethnicity Group", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)

    plt.tight_layout()
    return fig, ax


def plot_correlation_heatmap(df: pd.DataFrame, ax=None):
    """Pearson correlation heatmap for score columns.

    Args:
        df: DataFrame with numeric score columns.
        ax: Optional matplotlib Axes.

    Returns:
        matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    score_cols = ["math score", "reading score", "writing score"]
    if "Total Score" in df.columns:
        score_cols.extend(["Total Score", "Average Score"])

    corr = df[score_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, vmin=-1, vmax=1, square=True)
    ax.set_title("Score Correlation Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig, ax


def save_plot(fig, filename: str, output_dir: str = None):
    """Save a matplotlib figure to the visuals/ directory.

    Args:
        fig: matplotlib Figure object.
        filename: Output filename (e.g. 'distributions.png').
        output_dir: Directory path. Defaults to visuals/ in project root.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "visuals")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {filepath}")
