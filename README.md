# 🛋️ FurnRent Analytics Intelligence Dashboard

A comprehensive data-driven analytics platform for a UAE-based furniture rental business.
Built with Streamlit — deployable in one click on Streamlit Community Cloud.

## 📊 Features

| Tab | Analysis Type | Methods |
|-----|--------------|---------|
| Descriptive | What is happening | Charts, KPIs, distributions |
| Diagnostic | Why is it happening | Correlation, Chi-square, leakage |
| Predictive Models | What will happen | Logistic Regression, Random Forest, XGBoost |
| Customer Segments | Who are they | K-Means clustering, PCA, playbooks |
| Association Rules | What goes together | Apriori, Confidence, Lift |
| Predict New Customers | Future leads | Ensemble model + CSV upload |

## 🚀 Deploy on Streamlit Cloud

1. Fork / push this repo to GitHub (no sub-folders needed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Click **Deploy** — done!

## 🛠️ Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Files

```
app.py                              ← Main Streamlit application
requirements.txt                    ← All Python dependencies
furniture_rental_survey_cleaned.csv ← Training data (2,000 respondents)
.streamlit/config.toml              ← Theme configuration
README.md                           ← This file
```

## 📈 Algorithms Used

- **Classification**: Logistic Regression, Random Forest, XGBoost
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **Explainability**: SHAP values (feature importance)
- **Clustering**: K-Means (4 segments), PCA for 2D visualization
- **Association Mining**: Apriori algorithm — Support, Confidence, Lift
- **Class Imbalance**: SMOTE oversampling

## 🎯 Target Variable

`will_subscribe` — binary (0/1) indicating subscription intent
- 1,301 positive (65.05%)
- 699 negative (34.95%)
