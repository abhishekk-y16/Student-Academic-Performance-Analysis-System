# Student Academic Performance Analysis System

A comprehensive data analytics system that explores the relationship between demographic, socioeconomic factors and student academic performance. Built with Python, featuring interactive visualizations and predictive modeling.

## Dataset

**Students Performance in Exams** (Kaggle) — 1,000 student records across 8 dimensions:
- **Categorical**: gender, race/ethnicity, parental level of education, lunch type, test preparation course
- **Continuous**: math score, reading score, writing score (0–100)

## Project Structure

```
├── data/                   # Raw dataset (StudentsPerformance.csv)
├── notebooks/              # Jupyter EDA notebook (analysis.ipynb)
├── src/                    # Core Python modules
│   ├── data_loader.py      # Data ingestion & validation
│   ├── feature_engineering.py  # Composite features & encoding
│   ├── analysis.py         # Reusable EDA plotting functions
│   └── ml_model.py         # ML training, evaluation & prediction
├── dashboard/              # Streamlit interactive dashboard (app.py)
├── visuals/                # Exported PNG plots
├── models/                 # Serialized ML models (.pkl)
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
# Ensure data/StudentsPerformance.csv exists

# 3. Train ML models (required for dashboard prediction tab)
python -c "from src.ml_model import train_and_save_models; from src.data_loader import load_data; from src.feature_engineering import add_composite_scores; train_and_save_models(add_composite_scores(load_data()))"
```

## Usage

### Run the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### Run the Jupyter Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Key Features

- **Data Quality Assurance**: Automated null/duplicate detection and dtype validation
- **Feature Engineering**: Total Score, Average Score, Pass/Fail flags, letter grades
- **Exploratory Data Analysis**: Score distributions, subject comparisons, demographic breakdowns
- **Correlation Mapping**: Pearson correlation heatmaps with multicollinearity detection
- **Interactive Dashboard**: Streamlit app with cross-filtering, KPIs, and 4 analytical tabs
- **Predictive Modeling**: Logistic Regression & Random Forest for Pass/Fail classification; Linear Regression & Random Forest for score prediction

## Key Insights

1. **Mathematics** is the most challenging subject (lowest median, widest variance)
2. **Test preparation courses** yield ~10% average score improvement
3. **Parental education** and **lunch type** (SES proxy) are strong performance predictors
4. **Reading & Writing** are highly correlated (r > 0.90), while Math is more independent
5. **Gender gaps** exist: females outperform in reading/writing, males show marginal math advantage
