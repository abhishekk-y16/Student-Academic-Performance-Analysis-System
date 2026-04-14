"""Student Academic Performance Dashboard — Streamlit App."""

import sys
import os

# Add project root to path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from src.data_loader import load_data, validate_data
from src.feature_engineering import add_composite_scores

#  Page Configuration 
st.set_page_config(
    page_title="Student Academic Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


#Data Loading (cached) 
@st.cache_data
def load_and_prepare_data():
    """Load dataset, validate, and add Engineered features."""
    df = load_data()
    # Ensure score columns are numeric
    for col in ["math score", "reading score", "writing score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    df = add_composite_scores(df)
    return df


df_full = load_and_prepare_data()

#  Sidebar Filters 
st.sidebar.title("🎛️ Filters")
st.sidebar.markdown("Use Filters to explore Specific student subgroups.")

selected_genders = st.sidebar.multiselect(
    "Gender",
    options=sorted(df_full["gender"].unique()),
    default=sorted(df_full["gender"].unique()),
)

selected_races = st.sidebar.multiselect(
    "Race/Ethnicity",
    options=sorted(df_full["race/ethnicity"].unique()),
    default=sorted(df_full["race/ethnicity"].unique()),
)

edu_order = [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree",
]
available_edu = [e for e in edu_order if e in df_full["parental level of education"].unique()]
selected_edu = st.sidebar.multiselect(
    "Parental Education",
    options=available_edu,
    default=available_edu,
)

selected_lunch = st.sidebar.multiselect(
    "Lunch Type",
    options=sorted(df_full["lunch"].unique()),
    default=sorted(df_full["lunch"].unique()),
)

selected_prep = st.sidebar.multiselect(
    "Test Preparation",
    options=sorted(df_full["test preparation course"].unique()),
    default=sorted(df_full["test preparation course"].unique()),
)

score_range = st.sidebar.slider(
    "Average Score Range",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=1.0,
)

# Apply filters
filtered_df = df_full[
    (df_full["gender"].isin(selected_genders))
    & (df_full["race/ethnicity"].isin(selected_races))
    & (df_full["parental level of education"].isin(selected_edu))
    & (df_full["lunch"].isin(selected_lunch))
    & (df_full["test preparation course"].isin(selected_prep))
    & (df_full["Average Score"] >= score_range[0])
    & (df_full["Average Score"] <= score_range[1])
]

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Students", f"{len(filtered_df):,}", f"{len(filtered_df) - len(df_full):+,} vs total")

# ── Title ────────────────────────────────────────────────────────────────────
st.title("📊 Student Academic Performance Dashboard")
st.markdown("Interactive analysis of **1,000 students** across demographic, socioeconomic, and academic dimensions.")

# ── KPI Metrics Row ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Students",
        f"{len(filtered_df):,}",
        f"{len(filtered_df) - len(df_full):+,}",
    )
with col2:
    avg_math = filtered_df["math score"].mean()
    full_math = df_full["math score"].mean()
    st.metric("Avg Math Score", f"{avg_math:.1f}", f"{avg_math - full_math:+.1f}")
with col3:
    avg_read = filtered_df["reading score"].mean()
    full_read = df_full["reading score"].mean()
    st.metric("Avg Reading Score", f"{avg_read:.1f}", f"{avg_read - full_read:+.1f}")
with col4:
    avg_write = filtered_df["writing score"].mean()
    full_write = df_full["writing score"].mean()
    st.metric("Avg Writing Score", f"{avg_write:.1f}", f"{avg_write - full_write:+.1f}")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Score Distributions",
    "👥 Demographic Comparisons",
    "🔗 Correlation Analysis",
    "🤖 Predictive Model",
])

# ── Tab 1: Score Distributions ──────────────────────────────────────────────
with tab1:
    st.subheader("Interactive Score Distribution")

    subject_choice = st.selectbox(
        "Select Subject",
        ["math score", "reading score", "writing score"],
        key="dist_subject",
    )

    fig_hist = px.histogram(
        filtered_df,
        x=subject_choice,
        nbins=20,
        marginal="box",
        color="gender",
        barmode="overlay",
        opacity=0.7,
        title=f"Distribution of {subject_choice.title()}",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_hist.update_layout(xaxis_title="Score", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Subject comparison box plots
    st.subheader("Subject-Wise Performance Comparison")
    score_melted = filtered_df[["math score", "reading score", "writing score"]].melt(
        var_name="Subject", value_name="Score"
    )
    fig_box = px.box(
        score_melted,
        x="Subject",
        y="Score",
        color="Subject",
        title="Comparative Box Plots Across Subjects",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ── Tab 2: Demographic Comparisons ──────────────────────────────────────────
with tab2:
    st.subheader("Performance Disparities by Preparation and Gender")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_prep = px.box(
            filtered_df,
            x="test preparation course",
            y="Average Score",
            color="test preparation course",
            title="Impact of Test Preparation Course",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_prep, use_container_width=True)

    with col_right:
        fig_gender = px.box(
            filtered_df,
            x="gender",
            y="Average Score",
            color="gender",
            title="Performance by Gender",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    # Parental Edu
    st.subheader("Impact of Parental Education Level")
    fig_edu = px.box(
        filtered_df,
        x="parental level of education",
        y="Average Score",
        color="parental level of education",
        title="Average Score by Parental Education Level",
        category_orders={"parental level of education": edu_order},
        color_discrete_sequence=px.colors.sequential.YlOrRd,
    )
    fig_edu.update_layout(showlegend=False)
    st.plotly_chart(fig_edu, use_container_width=True)

    # Lunch Type
    st.subheader("Impact of Lunch Type (Socioeconomic Proxy)")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        fig_lunch = px.box(
            filtered_df,
            x="lunch",
            y="Average Score",
            color="lunch",
            title="Average Score by Lunch Type",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(fig_lunch, use_container_width=True)

    with col_r2:
        fig_race = px.box(
            filtered_df,
            x="race/ethnicity",
            y="Average Score",
            color="race/ethnicity",
            title="Average Score by Race/Ethnicity",
            category_orders={"race/ethnicity": sorted(filtered_df["race/ethnicity"].unique())},
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        st.plotly_chart(fig_race, use_container_width=True)

# ── Tab 3: Correlation Analysis ─────────────────────────────────────────────
with tab3:
    st.subheader("Subject Correlation Matrix")

    score_cols = ["math score", "reading score", "writing score"]
    corr = filtered_df[score_cols].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="coolwarm",
        ax=ax_corr, linewidths=0.5, vmin=-1, vmax=1, square=True,
    )
    ax_corr.set_title("Pearson Correlation: Math, Reading, Writing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # Scatter matrix
    st.subheader("Pairwise Score Relationships")
    fig_scatter = px.scatter_matrix(
        filtered_df,
        dimensions=score_cols,
        color="gender",
        title="Score Scatter Matrix by Gender",
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=3))
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Tab 4: Predictive Model ─────────────────────────────────────────────────
with tab4:
    st.subheader("🤖 Student Performance Predictor")
    st.markdown("Enter a Student profile to Predict their Expected scores using Trained Random Forest models.")

    # Check if models exist
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models_available = os.path.exists(os.path.join(model_dir, "rf_classifier.pkl"))

    if not models_available:
        st.warning(
            "⚠️ ML models Not found. Train them first by running:\n\n"
            "```python\n"
            "python -c \"from src.ml_model import train_and_save_models; "
            "from src.data_loader import load_data; "
            "from src.feature_engineering import add_composite_scores; "
            "train_and_save_models(add_composite_scores(load_data()))\"\n"
            "```"
        )
    else:
        from src.ml_model import load_models, predict_scores

        @st.cache_resource
        def get_models():
            return load_models()

        models = get_models()

        # Input form
        with st.form("prediction_form"):
            fc1, fc2 = st.columns(2)

            with fc1:
                pred_gender = st.selectbox("Gender", ["female", "male"], key="pred_gender")
                pred_race = st.selectbox(
                    "Race/Ethnicity",
                    sorted(df_full["race/ethnicity"].unique()),
                    key="pred_race",
                )
                pred_edu = st.selectbox(
                    "Parental Education",
                    edu_order,
                    key="pred_edu",
                )

            with fc2:
                pred_lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"], key="pred_lunch")
                pred_prep = st.selectbox(
                    "Test Preparation",
                    ["completed", "none"],
                    key="pred_prep",
                )

            submitted = st.form_submit_button("🔮 Predict Scores", use_container_width=True)

        if submitted:
            profile = {
                "gender": pred_gender,
                "race/ethnicity": pred_race,
                "parental level of education": pred_edu,
                "lunch": pred_lunch,
                "test preparation course": pred_prep,
            }

            results = predict_scores(profile, models)

            st.markdown("### Predicted Results")
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("Math Score", f"{results['math score']:.1f}")
            with rc2:
                st.metric("Reading Score", f"{results['reading score']:.1f}")
            with rc3:
                st.metric("Writing Score", f"{results['writing score']:.1f}")
            with rc4:
                st.metric("Average Score", f"{results['Average Score']:.1f}")

            st.markdown(f"**Overall Prediction:** {results['Overall Pass']}")

            # Feature importance chart
            st.subheader("Feature Importance (Random Forest)")
            try:
                import joblib
                rf_model = models.get("math_score")
                if rf_model is not None:
                    feature_cols = models["feature_cols"]
                    importances = rf_model.feature_importances_
                    fi_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "Importance": importances,
                    }).sort_values("Importance", ascending=True)

                    fig_fi = px.bar(
                        fi_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Feature Importance for Math Score Prediction",
                        color="Importance",
                        color_continuous_scale="Viridis",
                    )
                    fig_fi.update_layout(height=400)
                    st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.info(f"Feature importance visualization unavailable: {e}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Student Academic Performance Analysis System | "
    "Data: Students Performance in Exams (Kaggle) | "
    "Built with Streamlit, Plotly & Seaborn"
    "</div>",
    unsafe_allow_html=True,
)
