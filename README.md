# Student Lifestyle & Academic Performance Analytics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

An end-to-end data science and machine learning project analyzing how daily time allocations—study hours, sleep duration, physical activity, and social interactions—impact undergraduate academic performance ($CGPA$) and mental workload ($Stress\_Level$).

---

## Project Overview

This repository features a complete data analytics pipeline applied to a dataset of 2,000 student survey responses collected between August 2023 and May 2024. The study combines descriptive statistical profiling, inferential hypothesis testing, machine learning pipelines, and constrained numerical optimization to evaluate lifestyle trade-offs and prescribe balanced student daily routines.

### Key Analytical Pillars

1. Exploratory Data Analysis (EDA): Statistical distributions, summary statistics, and time-budget constraint verification.
2. Inferential Statistics: One-Way ANOVA, Tukey HSD post-hoc tests, Welch's t-tests, and Chi-Square tests of independence.
3. Predictive Modeling: Linear and Ridge Regression for continuous GPA prediction alongside multi-class Logistic Regression and Random Forests for Stress Level classification.
4. Prescriptive Analytics: Sequential Least Squares Programming (SLSQP) constrained optimization to maximize GPA under healthy lifestyle bounds.

---

## Key Findings

- Zero-Sum Time Allocation: Every observation strictly satisfies a 24.0-hour daily sum constraint across five activity categories ($Hours_{Total} = 24.0$), creating direct competition between academic and personal activities.
- Primary GPA Determinant: Daily study hours serve as the single dominant predictor of academic achievement ($r = 0.73$), accounting for over 50% of CGPA variance.
- The High-Stress / High-Achievement Paradox: Students in the `High` stress tier maintain a significantly higher mean GPA ($3.26$) than those in `Moderate` ($3.02$) and `Low` ($2.82$) tiers ($F = 434.89, p < 0.001$). However, high performers achieve this by sacrificing sleep ($7.05$ hrs/day) and physical activity due to heavy study routines ($8.39$ hrs/day).
- Predictive Pipeline Accuracy:
  - GPA Regression: Ridge Regression achieves an $R^2 \approx 0.57$ on held-out test data.
  - Stress Classification: Logistic Regression achieves 83% overall classification accuracy across `Low`, `Moderate`, and `High` stress categories.
- Prescriptive Optimization: Under healthy daily sleep boundaries ($\ge 7.0$ hours), the optimization framework projects an expected optimal yield of 3.52 / 4.00 GPA with a daily schedule of 10.0 hours study, 7.0 hours sleep, and 7.0 hours split between physical and social activities.

---

## Repository Structure

```text
├── data/
│   └── student_lifestyle_dataset.csv     # Raw dataset (2,000 observations)
├── figures/
│   ├── eda_report_summary.png            # Multi-panel EDA visualizations
│   ├── inferential_summary.png           # Confidence interval pointplots
│   └── ml_validation_performance.png     # Regression scatter & confusion matrix
├── notebooks/
│   └── Student_Lifestyle_Analysis.ipynb  # End-to-end Jupyter Notebook
├── .gitignore
├── LICENSE
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
└── src/                                  # Optional analysis scripts
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/student-lifestyle-analytics.git
cd student-lifestyle-analytics
```

### 2. Create and Activate a Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage

Launch Jupyter Notebook to run or modify the analytics workflow:

```bash
jupyter notebook notebooks/Student_Lifestyle_Analysis.ipynb
```

---

## Tech Stack & Tools

- Data Manipulation: pandas, numpy
- Statistical Analysis: scipy.stats (ANOVA, Tukey HSD, Welch's t-test, Chi-Square)
- Machine Learning & Optimization: scikit-learn, scipy.optimize (SLSQP)
- Data Visualization: matplotlib, seaborn

---

## Challenges & Methodological Limitations

- Model Expected Value vs. Individual Peaks: The optimizer predicts an expected mean GPA of 3.52 / 4.00 under healthy constraints. While individual top performers in the dataset reach $4.00$, parametric linear models estimate population expected values rather than extreme individual outliers.
- Multicollinearity: The 24-hour total daily limit creates exact linear dependency among activity features. Regularized models (Ridge Regression) were implemented to ensure stable coefficient estimations.
- Observational / Self-Reported Data: Feature metrics measure self-reported activity quantity (hours) rather than quality (e.g., study efficiency or sleep quality).

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
