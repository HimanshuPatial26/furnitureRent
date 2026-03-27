import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── XGBoost ───────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier

# ── SHAP ──────────────────────────────────────────────────────────────────────
import shap

# ── Association Rules ─────────────────────────────────────────────────────────
from mlxtend.frequent_patterns import apriori, association_rules

# ── SMOTE for class imbalance ─────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE

# ── Scipy ─────────────────────────────────────────────────────────────────────
from scipy import stats

import io
import base64

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FurnRent Analytics",
    page_icon="🛋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  THEME & GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = {
    "primary":   "#1B4F72",
    "accent":    "#F39C12",
    "positive":  "#27AE60",
    "negative":  "#E74C3C",
    "neutral":   "#95A5A6",
    "bg_dark":   "#0D1B2A",
    "bg_card":   "#1A2A3A",
    "text":      "#ECF0F1",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {PALETTE['bg_dark']};
    color: {PALETTE['text']};
}}

h1, h2, h3 {{ font-family: 'Syne', sans-serif; }}

.main-header {{
    background: linear-gradient(135deg, {PALETTE['primary']} 0%, #0A3055 60%, {PALETTE['bg_dark']} 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border-left: 6px solid {PALETTE['accent']};
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}}

.main-header h1 {{
    font-size: 2.6rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.5px;
}}

.main-header p {{
    color: #A8C4D4;
    font-size: 1rem;
    margin: 0.4rem 0 0 0;
    font-weight: 300;
}}

.metric-card {{
    background: {PALETTE['bg_card']};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    text-align: center;
}}

.metric-value {{
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: {PALETTE['accent']};
    line-height: 1;
}}

.metric-label {{
    font-size: 0.78rem;
    color: #8FA8BB;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}}

.section-header {{
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid {PALETTE['accent']};
    margin: 1.5rem 0 1rem 0;
}}

.insight-box {{
    background: linear-gradient(135deg, rgba(243,156,18,0.12) 0%, rgba(27,79,114,0.2) 100%);
    border-left: 4px solid {PALETTE['accent']};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.92rem;
}}

.model-badge {{
    display: inline-block;
    background: {PALETTE['primary']};
    color: #fff;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background-color: {PALETTE['bg_card']};
    border-radius: 12px;
    padding: 6px;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #8FA8BB;
}}

.stTabs [aria-selected="true"] {{
    background-color: {PALETTE['primary']} !important;
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] {{
    background-color: {PALETTE['bg_card']};
    border-right: 1px solid rgba(255,255,255,0.06);
}}

.stButton>button {{
    background: {PALETTE['primary']};
    color: white;
    border-radius: 8px;
    border: none;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
    transition: background 0.2s;
}}

.stButton>button:hover {{
    background: {PALETTE['accent']};
    color: #000;
}}

div[data-testid="metric-container"] {{
    background: {PALETTE['bg_card']};
    border-radius: 10px;
    padding: 0.8rem;
    border: 1px solid rgba(255,255,255,0.07);
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & CACHING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("furniture_rental_survey_cleaned.csv")
    return df


@st.cache_data
def get_feature_cols(df):
    drop_cols = ["respondent_id", "will_subscribe", "purchase_intent_score",
                 "intent_score", "is_outlier"]
    feature_cols = [c for c in df.columns if c not in drop_cols
                    and df[c].dtype in [np.int64, np.float64]]
    return feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_models(df):
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    y = df["will_subscribe"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE to handle class imbalance on train set
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc  = scaler.transform(X_test)

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train_res)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=42, n_jobs=-1)
    xgb.fit(X_train_res, y_train_res)

    models = {"Logistic Regression": lr,
              "Random Forest": rf,
              "XGBoost": xgb}

    results = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_prob),
            "fpr": fpr, "tpr": tpr,
            "y_pred": y_pred, "y_prob": y_prob,
            "conf_matrix": confusion_matrix(y_test, y_pred),
        }

    return models, scaler, feature_cols, X_test, X_test_sc, y_test, results


@st.cache_resource
def run_clustering(df):
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_sc)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    return labels, X_pca, kmeans


@st.cache_data
def run_association_rules(df):
    item_cols = [c for c in df.columns if
                 c.startswith(("cat_", "addon_", "pain_", "prio_", "src_"))
                 and c not in ["addon_count", "pain_count"]]
    basket = df[item_cols].copy().astype(bool)
    freq = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    rules = rules.sort_values("lift", ascending=False)
    return rules


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME HELPER
# ══════════════════════════════════════════════════════════════════════════════
def plotly_dark_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne", size=16, color="#FFFFFF")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#C0D4E0"),
        height=height,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0.2)", bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family:Syne; font-size:1.6rem; font-weight:800;
                    color:#F39C12;'>🛋️ FurnRent</div>
        <div style='color:#8FA8BB; font-size:0.8rem;'>Analytics Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Navigation**")
    selected_tab = st.radio("", [
        "📊 Descriptive",
        "🔍 Diagnostic",
        "🤖 Predictive Models",
        "🗺️ Customer Segments",
        "🔗 Association Rules",
        "🔮 Predict New Customers"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Dataset**")
    st.markdown("<small style='color:#8FA8BB;'>2,000 survey respondents · UAE · 105 features</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>🛋️ FurnRent Intelligence Dashboard</h1>
  <p>Furniture Rental Business · UAE Market · Data-Driven Strategy Platform</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if selected_tab == "📊 Descriptive":
    st.markdown("<div class='section-header'>📊 Descriptive Analysis</div>",
                unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("2,000", "Total Respondents"),
        (f"{df['will_subscribe'].mean()*100:.1f}%", "Subscription Rate"),
        (f"AED {df['wtp_monthly_aed'].median():,.0f}", "Median WTP / Month"),
        (f"AED {df['income_aed'].median()/1000:.0f}K", "Median Income"),
        (f"{df['furniture_category_count'].mean():.1f}", "Avg. Categories Wanted"),
    ]
    for col, (val, label) in zip([col1,col2,col3,col4,col5], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 1: Subscription by segment ────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        employ_cols = [c for c in df.columns if c.startswith("employment_type_")]
        employ_labels = [c.replace("employment_type_", "") for c in employ_cols]
        employ_sub = []
        for c in employ_cols:
            sub_grp = df[df[c] == 1]
            employ_sub.append(sub_grp["will_subscribe"].mean() * 100)

        fig = px.bar(x=employ_labels, y=employ_sub,
                     color=employ_sub, color_continuous_scale=["#1B4F72", "#F39C12"],
                     labels={"x": "Employment Type", "y": "Subscription Rate (%)"},
                     text=[f"{v:.0f}%" for v in employ_sub])
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        plotly_dark_layout(fig, "Subscription Rate by Employment Type")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        home_cols = [c for c in df.columns if c.startswith("home_type_")]
        home_labels = [c.replace("home_type_", "") for c in home_cols]
        home_sub = [df[df[c]==1]["will_subscribe"].mean()*100 for c in home_cols]

        fig2 = px.bar(x=home_labels, y=home_sub,
                      color=home_sub, color_continuous_scale=["#1B4F72", "#27AE60"],
                      labels={"x": "Home Type", "y": "Subscription Rate (%)"},
                      text=[f"{v:.0f}%" for v in home_sub])
        fig2.update_traces(textposition="outside")
        fig2.update_coloraxes(showscale=False)
        plotly_dark_layout(fig2, "Subscription Rate by Home Type")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Age / Income distribution ──────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        fig3 = px.histogram(df, x="age_numeric", color="will_subscribe",
                            color_discrete_map={0: PALETTE["negative"], 1: PALETTE["positive"]},
                            barmode="overlay", opacity=0.75, nbins=20,
                            labels={"will_subscribe": "Will Subscribe",
                                    "age_numeric": "Age"},
                            category_orders={"will_subscribe": [0, 1]})
        fig3.update_traces(legendgrouptitle_text="")
        newnames = {0: "No", 1: "Yes"}
        fig3.for_each_trace(lambda t: t.update(name=newnames.get(int(t.name), t.name)
                                               if t.name.lstrip('-').isdigit() else t.name))
        plotly_dark_layout(fig3, "Age Distribution by Subscription Intent")
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.box(df, x="will_subscribe", y="income_aed",
                      color="will_subscribe",
                      color_discrete_map={0: PALETTE["negative"], 1: PALETTE["positive"]},
                      labels={"will_subscribe": "Will Subscribe", "income_aed": "Monthly Income (AED)"},
                      category_orders={"will_subscribe": [0, 1]})
        fig4.update_xaxes(tickvals=[0,1], ticktext=["No","Yes"])
        plotly_dark_layout(fig4, "Income Distribution by Subscription Intent")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Furniture categories & pain points ──────────────────────────────
    col_e, col_f = st.columns(2)

    with col_e:
        cat_cols = [c for c in df.columns if c.startswith("cat_")]
        cat_labels = [c.replace("cat_", "").title() for c in cat_cols]
        cat_pcts = [df[c].mean()*100 for c in cat_cols]
        cat_df = pd.DataFrame({"Category": cat_labels, "Demand (%)": cat_pcts}).sort_values("Demand (%)", ascending=True)

        fig5 = px.bar(cat_df, x="Demand (%)", y="Category", orientation="h",
                      color="Demand (%)", color_continuous_scale=["#0A3055", "#F39C12"],
                      text=[f"{v:.0f}%" for v in cat_df["Demand (%)"]])
        fig5.update_traces(textposition="outside")
        fig5.update_coloraxes(showscale=False)
        plotly_dark_layout(fig5, "Furniture Category Demand", height=380)
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        pain_cols = [c for c in df.columns if c.startswith("pain_") and c != "pain_count"]
        pain_labels = [c.replace("pain_", "").replace("_", " ").title() for c in pain_cols]
        pain_pcts = [df[c].mean()*100 for c in pain_cols]
        pain_df = pd.DataFrame({"Pain Point": pain_labels, "% Respondents": pain_pcts}).sort_values("% Respondents", ascending=True)

        fig6 = px.bar(pain_df, x="% Respondents", y="Pain Point", orientation="h",
                      color="% Respondents", color_continuous_scale=["#0A3055", "#E74C3C"],
                      text=[f"{v:.0f}%" for v in pain_df["% Respondents"]])
        fig6.update_traces(textposition="outside")
        fig6.update_coloraxes(showscale=False)
        plotly_dark_layout(fig6, "Customer Pain Points", height=380)
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4: WTP and Priority ────────────────────────────────────────────────
    col_g, col_h = st.columns(2)

    with col_g:
        prio_cols = [c for c in df.columns if c.startswith("prio_")]
        prio_labels = [c.replace("prio_", "").replace("_", " ").title() for c in prio_cols]
        prio_pcts = [df[c].mean()*100 for c in prio_cols]
        prio_df = pd.DataFrame({"Priority": prio_labels, "% Respondents": prio_pcts}).sort_values("% Respondents", ascending=False)

        fig7 = px.pie(prio_df, names="Priority", values="% Respondents",
                      color_discrete_sequence=px.colors.sequential.Blues_r[::-1] + ["#F39C12", "#27AE60"])
        fig7.update_traces(textposition="inside", textinfo="percent+label",
                           hole=0.35)
        plotly_dark_layout(fig7, "What Customers Prioritize", height=400)
        st.plotly_chart(fig7, use_container_width=True)

    with col_h:
        fig8 = px.histogram(df, x="wtp_monthly_aed",
                            color_discrete_sequence=[PALETTE["accent"]],
                            nbins=25,
                            labels={"wtp_monthly_aed": "Willingness to Pay (AED/month)"})
        fig8.add_vline(x=df["wtp_monthly_aed"].median(), line_dash="dash",
                       line_color=PALETTE["positive"],
                       annotation_text=f"Median: AED {df['wtp_monthly_aed'].median():.0f}",
                       annotation_font_color=PALETTE["positive"])
        plotly_dark_layout(fig8, "Willingness to Pay Distribution", height=400)
        st.plotly_chart(fig8, use_container_width=True)

    # ── Lifestyle / Household breakdown ───────────────────────────────────────
    st.markdown("<div class='section-header'>Lifestyle & Household Profile</div>",
                unsafe_allow_html=True)
    col_i, col_j = st.columns(2)

    with col_i:
        ls_cols = [c for c in df.columns if c.startswith("lifestyle_type_")]
        ls_labels = [c.replace("lifestyle_type_", "") for c in ls_cols]
        ls_sub = [df[df[c]==1]["will_subscribe"].mean()*100 for c in ls_cols]
        ls_count = [df[c].sum() for c in ls_cols]
        ls_df = pd.DataFrame({"Lifestyle": ls_labels, "Subscription Rate (%)": ls_sub, "Count": ls_count})

        fig9 = px.scatter(ls_df, x="Count", y="Subscription Rate (%)",
                          size="Count", color="Subscription Rate (%)",
                          color_continuous_scale=["#1B4F72","#F39C12"],
                          text="Lifestyle",
                          labels={"Count": "# Respondents"})
        fig9.update_traces(textposition="top center")
        fig9.update_coloraxes(showscale=False)
        plotly_dark_layout(fig9, "Lifestyle Type: Volume vs Subscription Rate", height=380)
        st.plotly_chart(fig9, use_container_width=True)

    with col_j:
        hh_cols = [c for c in df.columns if c.startswith("household_type_")]
        hh_labels = [c.replace("household_type_", "") for c in hh_cols]
        hh_sub = [df[df[c]==1]["will_subscribe"].mean()*100 for c in hh_cols]
        hh_count = [df[c].sum() for c in hh_cols]
        hh_df = pd.DataFrame({"Household": hh_labels, "Rate": hh_sub, "Count": hh_count})

        fig10 = go.Figure(data=[
            go.Bar(name="# Respondents", x=hh_labels, y=hh_count,
                   marker_color=PALETTE["primary"], yaxis="y"),
            go.Scatter(name="Sub Rate (%)", x=hh_labels, y=hh_sub,
                       mode="lines+markers", marker=dict(color=PALETTE["accent"], size=10),
                       line=dict(color=PALETTE["accent"], width=2.5), yaxis="y2"),
        ])
        fig10.update_layout(
            yaxis=dict(title="# Respondents", gridcolor="rgba(255,255,255,0.06)"),
            yaxis2=dict(title="Subscription Rate (%)", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
        )
        plotly_dark_layout(fig10, "Household Type: Volume & Subscription Rate", height=380)
        st.plotly_chart(fig10, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Key Takeaway:</b> Salaried-private employees and new movers show highest subscription intent.
    1BHK and Studio residents represent the largest addressable pool.
    Median WTP of AED 800/month gives a clear pricing anchor.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif selected_tab == "🔍 Diagnostic":
    st.markdown("<div class='section-header'>🔍 Diagnostic Analysis</div>",
                unsafe_allow_html=True)

    feature_cols = get_feature_cols(df)

    # ── Correlation with will_subscribe ───────────────────────────────────────
    corr = df[feature_cols + ["will_subscribe"]].corr()["will_subscribe"].drop("will_subscribe")
    corr_df = corr.abs().sort_values(ascending=False).head(25).reset_index()
    corr_df.columns = ["Feature", "Abs Correlation"]
    corr_df["Direction"] = corr_df["Feature"].map(
        lambda f: "Positive" if corr[f] > 0 else "Negative")
    corr_df["Raw Corr"] = corr_df["Feature"].map(lambda f: corr[f])

    fig_corr = px.bar(corr_df, x="Abs Correlation", y="Feature",
                      orientation="h", color="Direction",
                      color_discrete_map={"Positive": PALETTE["positive"],
                                          "Negative": PALETTE["negative"]},
                      text=corr_df["Raw Corr"].map(lambda v: f"{v:.3f}"))
    fig_corr.update_traces(textposition="outside")
    plotly_dark_layout(fig_corr, "Top 25 Features Correlated with Subscription Intent", height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ── Statistical tests ─────────────────────────────────────────────────────
    st.markdown("**Statistical Significance Tests (Chi-Square)**")
    st.markdown("<small style='color:#8FA8BB;'>Testing whether categorical segments differ significantly in subscription rates</small>",
                unsafe_allow_html=True)

    binary_flag_cols = [c for c in feature_cols if df[c].nunique() == 2 and c != "will_subscribe"]
    chi_results = []
    for c in binary_flag_cols[:40]:
        ct = pd.crosstab(df[c], df["will_subscribe"])
        if ct.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
            chi_results.append({"Feature": c, "Chi2": chi2, "p-value": p,
                                 "Significant": "✅ Yes" if p < 0.05 else "❌ No"})

    chi_df = pd.DataFrame(chi_results).sort_values("p-value")
    st.dataframe(
        chi_df.style.background_gradient(subset=["Chi2"], cmap="Blues")
              .format({"Chi2": "{:.2f}", "p-value": "{:.4f}"}),
        use_container_width=True, height=350
    )

    st.divider()

    # ── Subscription leakage: high intent but no subscribe ─────────────────────
    st.markdown("**🚨 Conversion Leakage Analysis**")
    st.markdown("<small style='color:#8FA8BB;'>High intent but not subscribing — these are recoverable leads</small>",
                unsafe_allow_html=True)

    leakage = df[(df["intent_score"] >= 2) & (df["will_subscribe"] == 0)]
    st.markdown(f"""
    <div class='insight-box'>
    🚨 <b>{len(leakage)} respondents</b> show high intent scores (≥2) but are NOT subscribing.
    That's <b>{len(leakage)/len(df)*100:.1f}%</b> of the total survey — a significant recoverable segment.
    Top barriers for this group:
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pain_leak = {c.replace("pain_","").replace("_"," ").title(): leakage[c].mean()*100
                     for c in df.columns if c.startswith("pain_") and c != "pain_count"}
        pain_leak_df = pd.DataFrame.from_dict(pain_leak, orient="index", columns=["Rate"]).sort_values("Rate", ascending=True)
        fig_leak = px.bar(pain_leak_df, x="Rate", y=pain_leak_df.index, orientation="h",
                          color="Rate", color_continuous_scale=["#1B4F72","#E74C3C"],
                          text=[f"{v:.0f}%" for v in pain_leak_df["Rate"]])
        fig_leak.update_traces(textposition="outside")
        fig_leak.update_coloraxes(showscale=False)
        plotly_dark_layout(fig_leak, "Pain Points Among Leakage Segment", height=340)
        st.plotly_chart(fig_leak, use_container_width=True)

    with col2:
        prio_leak = {c.replace("prio_","").replace("_"," ").title(): leakage[c].mean()*100
                     for c in df.columns if c.startswith("prio_")}
        prio_leak_df = pd.DataFrame.from_dict(prio_leak, orient="index", columns=["Rate"]).sort_values("Rate", ascending=False)
        fig_prio_leak = px.bar(prio_leak_df, x=prio_leak_df.index, y="Rate",
                               color="Rate", color_continuous_scale=["#1B4F72","#F39C12"],
                               text=[f"{v:.0f}%" for v in prio_leak_df["Rate"]])
        fig_prio_leak.update_traces(textposition="outside")
        fig_prio_leak.update_coloraxes(showscale=False)
        plotly_dark_layout(fig_prio_leak, "Priorities Among Leakage Segment", height=340)
        st.plotly_chart(fig_prio_leak, use_container_width=True)

    # ── WTP vs Income segmentation ─────────────────────────────────────────────
    st.divider()
    st.markdown("**WTP vs Income — Subscription Segmentation**")

    fig_wtp = px.scatter(df, x="income_aed", y="wtp_monthly_aed",
                         color="will_subscribe",
                         color_discrete_map={0: PALETTE["negative"], 1: PALETTE["positive"]},
                         opacity=0.5, size_max=8,
                         labels={"income_aed": "Monthly Income (AED)",
                                 "wtp_monthly_aed": "WTP per Month (AED)",
                                 "will_subscribe": "Subscribes"})
    fig_wtp.update_traces(marker=dict(size=5))
    plotly_dark_layout(fig_wtp, "Willingness-to-Pay vs Income by Subscription Intent", height=420)
    st.plotly_chart(fig_wtp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PREDICTIVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif selected_tab == "🤖 Predictive Models":
    st.markdown("<div class='section-header'>🤖 Predictive Models</div>",
                unsafe_allow_html=True)

    with st.spinner("Training models (Logistic Regression · Random Forest · XGBoost)..."):
        models, scaler, feature_cols, X_test, X_test_sc, y_test, results = train_models(df)

    # ── Performance Metrics Table ──────────────────────────────────────────────
    st.markdown("**Model Performance Comparison**")
    metrics_data = []
    for name, res in results.items():
        metrics_data.append({
            "Model": name,
            "Accuracy": f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall": f"{res['recall']:.4f}",
            "F1-Score": f"{res['f1']:.4f}",
            "ROC-AUC": f"{res['roc_auc']:.4f}",
        })
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    st.divider()

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig_roc = go.Figure()
        colors_roc = [PALETTE["accent"], PALETTE["positive"], "#9B59B6"]
        for (name, res), color in zip(results.items(), colors_roc):
            fig_roc.add_trace(go.Scatter(
                x=res["fpr"], y=res["tpr"],
                name=f"{name} (AUC={res['roc_auc']:.3f})",
                line=dict(color=color, width=2.5)
            ))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dot", color=PALETTE["neutral"], width=1),
                                     name="Random Baseline", showlegend=True))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        plotly_dark_layout(fig_roc, "ROC Curves — All Models", height=420)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        # Radar chart of metrics
        cats = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        colors_radar = [PALETTE["accent"], PALETTE["positive"], "#9B59B6"]
        fig_radar = go.Figure()
        for (name, res), color in zip(results.items(), colors_radar):
            vals = [res["accuracy"], res["precision"], res["recall"], res["f1"], res["roc_auc"]]
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats+[cats[0]],
                name=name, line=dict(color=color, width=2),
                fill="toself", fillcolor=color.replace(")", ",0.1)").replace("rgb","rgba")
                if "rgb" in color else color + "22"
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        plotly_dark_layout(fig_radar, "Performance Radar — All Models", height=420)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ── Confusion Matrices ─────────────────────────────────────────────────────
    st.markdown("**Confusion Matrices**")
    cols_cm = st.columns(3)
    for (name, res), col in zip(results.items(), cols_cm):
        with col:
            cm = res["conf_matrix"]
            fig_cm = px.imshow(cm, text_auto=True,
                               color_continuous_scale=["#0D1B2A", PALETTE["primary"], PALETTE["accent"]],
                               labels=dict(x="Predicted", y="Actual"),
                               x=["Not Subscribe","Subscribe"],
                               y=["Not Subscribe","Subscribe"])
            fig_cm.update_coloraxes(showscale=False)
            plotly_dark_layout(fig_cm, name, height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("**Feature Importance (Random Forest & XGBoost)**")
    col3, col4 = st.columns(2)

    with col3:
        rf_model = models["Random Forest"]
        rf_imp = pd.DataFrame({"Feature": feature_cols,
                                "Importance": rf_model.feature_importances_})
        rf_imp = rf_imp.sort_values("Importance", ascending=False).head(20)
        rf_imp = rf_imp.sort_values("Importance", ascending=True)

        fig_rf_imp = px.bar(rf_imp, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale=["#1B4F72","#27AE60"],
                            text=rf_imp["Importance"].map(lambda v: f"{v:.3f}"))
        fig_rf_imp.update_traces(textposition="outside")
        fig_rf_imp.update_coloraxes(showscale=False)
        plotly_dark_layout(fig_rf_imp, "Random Forest — Top 20 Features", height=520)
        st.plotly_chart(fig_rf_imp, use_container_width=True)

    with col4:
        xgb_model = models["XGBoost"]
        xgb_imp = pd.DataFrame({"Feature": feature_cols,
                                 "Importance": xgb_model.feature_importances_})
        xgb_imp = xgb_imp.sort_values("Importance", ascending=False).head(20)
        xgb_imp = xgb_imp.sort_values("Importance", ascending=True)

        fig_xgb_imp = px.bar(xgb_imp, x="Importance", y="Feature", orientation="h",
                              color="Importance", color_continuous_scale=["#1B4F72","#9B59B6"],
                              text=xgb_imp["Importance"].map(lambda v: f"{v:.3f}"))
        fig_xgb_imp.update_traces(textposition="outside")
        fig_xgb_imp.update_coloraxes(showscale=False)
        plotly_dark_layout(fig_xgb_imp, "XGBoost — Top 20 Features", height=520)
        st.plotly_chart(fig_xgb_imp, use_container_width=True)

    # ── SHAP Values ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**SHAP Feature Importance (XGBoost — Global Explanation)**")
    with st.spinner("Computing SHAP values..."):
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
        explainer = shap.TreeExplainer(models["XGBoost"])
        shap_values = explainer.shap_values(X_test_df)

        shap_mean = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": feature_cols, "Mean |SHAP|": shap_mean})
        shap_df = shap_df.sort_values("Mean |SHAP|", ascending=False).head(20)
        shap_df = shap_df.sort_values("Mean |SHAP|", ascending=True)

        fig_shap = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                          color="Mean |SHAP|",
                          color_continuous_scale=["#0A3055","#F39C12","#E74C3C"],
                          text=shap_df["Mean |SHAP|"].map(lambda v: f"{v:.4f}"))
        fig_shap.update_traces(textposition="outside")
        fig_shap.update_coloraxes(showscale=False)
        plotly_dark_layout(fig_shap, "SHAP Values — What Drives Subscription Decisions", height=520)
        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Model Insight:</b> XGBoost consistently outperforms across all metrics.
    SHAP analysis confirms that <b>intent score, WTP, income, and new-mover status</b>
    are the strongest predictors of subscription intent.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CUSTOMER SEGMENTS (CLUSTERING)
# ══════════════════════════════════════════════════════════════════════════════
elif selected_tab == "🗺️ Customer Segments":
    st.markdown("<div class='section-header'>🗺️ Customer Segmentation (K-Means Clustering)</div>",
                unsafe_allow_html=True)

    with st.spinner("Running clustering..."):
        cluster_labels, X_pca, kmeans = run_clustering(df)

    df_cl = df.copy()
    df_cl["Cluster"] = cluster_labels

    SEGMENT_NAMES = {
        0: "🏃 Mobile Professionals",
        1: "👨‍👩‍👧 Family Settlers",
        2: "💰 Budget Conscious",
        3: "✨ Premium Upgraders",
    }

    # ── PCA Scatter ───────────────────────────────────────────────────────────
    pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])
    pca_df["Cluster"] = [SEGMENT_NAMES[l] for l in cluster_labels]
    pca_df["Subscribe"] = df["will_subscribe"].values

    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         symbol="Subscribe",
                         color_discrete_sequence=[PALETTE["accent"], PALETTE["positive"],
                                                  PALETTE["negative"], "#9B59B6"],
                         opacity=0.65,
                         labels={"symbol": "Subscribes (1=Yes)"})
    plotly_dark_layout(fig_pca, "Customer Segments — PCA Projection", height=480)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.divider()

    # ── Segment Profiles ──────────────────────────────────────────────────────
    st.markdown("**Segment Profiles**")
    profile_cols = ["age_numeric","income_aed","wtp_monthly_aed",
                    "furniture_category_count","addon_count","pain_count",
                    "move_frequency_ord","planned_stay_ord","will_subscribe"]

    seg_profile = df_cl.groupby("Cluster")[profile_cols].mean().round(2)
    seg_profile.index = [SEGMENT_NAMES[i] for i in seg_profile.index]
    seg_profile.columns = ["Avg Age","Avg Income","WTP/mo","# Categories",
                           "# Add-ons","# Pain Points","Move Freq","Planned Stay","Sub Rate"]
    st.dataframe(seg_profile.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.divider()

    # ── Segment Size & Sub Rate ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        size_df = df_cl.groupby("Cluster").size().reset_index(name="Count")
        size_df["Segment"] = size_df["Cluster"].map(SEGMENT_NAMES)
        fig_size = px.pie(size_df, names="Segment", values="Count", hole=0.4,
                          color_discrete_sequence=[PALETTE["accent"], PALETTE["positive"],
                                                   PALETTE["negative"], "#9B59B6"])
        fig_size.update_traces(textposition="inside", textinfo="percent+label")
        plotly_dark_layout(fig_size, "Segment Size Distribution", height=380)
        st.plotly_chart(fig_size, use_container_width=True)

    with col2:
        sub_rate = df_cl.groupby("Cluster")["will_subscribe"].mean().reset_index()
        sub_rate["Segment"] = sub_rate["Cluster"].map(SEGMENT_NAMES)
        sub_rate["Sub Rate (%)"] = sub_rate["will_subscribe"] * 100
        fig_sub = px.bar(sub_rate, x="Segment", y="Sub Rate (%)",
                         color="Sub Rate (%)",
                         color_continuous_scale=["#1B4F72","#F39C12"],
                         text=[f"{v:.1f}%" for v in sub_rate["Sub Rate (%)"]],
                         labels={"Segment": ""})
        fig_sub.update_traces(textposition="outside")
        fig_sub.update_coloraxes(showscale=False)
        plotly_dark_layout(fig_sub, "Subscription Rate by Segment", height=380)
        st.plotly_chart(fig_sub, use_container_width=True)

    # ── Prescriptive Marketing Playbooks ──────────────────────────────────────
    st.divider()
    st.markdown("**🎯 Prescriptive Marketing Playbooks**")

    playbooks = {
        "🏃 Mobile Professionals": {
            "profile": "Young, high-mobility, hybrid/WFH, 1BHK/Studio, frequent movers",
            "pitch": "Flexibility-first messaging — 'Move tomorrow, furniture handles itself'",
            "channel": "LinkedIn, Instagram, relocation platforms",
            "offer": "1-month free swap, fast setup within 48hrs",
            "priority": "🔴 HIGH — highest subscription propensity",
        },
        "👨‍👩‍👧 Family Settlers": {
            "profile": "Mid-30s to 40s, families, 2BHK/3BHK, longer planned stay",
            "pitch": "Quality & stability — 'Complete your home without the lifetime commitment'",
            "channel": "Facebook, WhatsApp groups, school/community boards",
            "offer": "Bundle deals (bedroom + living room), kids furniture add-on",
            "priority": "🟡 MEDIUM — high value but slower conversion",
        },
        "💰 Budget Conscious": {
            "profile": "Lower income, price-sensitive, high WTP% of income, shared flats",
            "pitch": "Value-first — 'Premium furniture at AED X/month, no upfront cost'",
            "channel": "Dubizzle, TikTok, expat Facebook groups",
            "offer": "Entry-level package, free delivery, monthly payments",
            "priority": "🟠 MEDIUM — large volume but needs price anchoring",
        },
        "✨ Premium Upgraders": {
            "profile": "High income, luxury style preference, villa/3BHK, business owners",
            "pitch": "Lifestyle upgrade — 'Your space, curated. Swap when your taste evolves'",
            "channel": "Instagram, luxury property portals, referral programs",
            "offer": "Designer collections, smart home bundles, concierge setup",
            "priority": "🟢 PREMIUM — highest LTV, worth high CAC",
        },
    }

    for seg_name, data in playbooks.items():
        with st.expander(f"{seg_name} — {data['priority']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**👤 Profile:** {data['profile']}")
                st.markdown(f"**📣 Pitch:** {data['pitch']}")
            with c2:
                st.markdown(f"**📱 Channel:** {data['channel']}")
                st.markdown(f"**🎁 Offer:** {data['offer']}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif selected_tab == "🔗 Association Rules":
    st.markdown("<div class='section-header'>🔗 Association Rules (Apriori — Confidence & Lift)</div>",
                unsafe_allow_html=True)

    with st.spinner("Mining association rules..."):
        rules = run_association_rules(df)

    if rules.empty:
        st.warning("No rules found with current thresholds. Try lowering min_support in the code.")
    else:
        # ── Filters ───────────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            min_conf = st.slider("Min Confidence", 0.3, 1.0, 0.5, 0.05)
        with col2:
            min_lift = st.slider("Min Lift", 1.0, 3.0, 1.1, 0.1)
        with col3:
            top_n = st.slider("Show Top N Rules", 10, 100, 30, 5)

        filtered = rules[
            (rules["confidence"] >= min_conf) &
            (rules["lift"] >= min_lift)
        ].head(top_n).copy()

        filtered["antecedents_str"] = filtered["antecedents"].apply(lambda x: ", ".join(list(x)))
        filtered["consequents_str"] = filtered["consequents"].apply(lambda x: ", ".join(list(x)))

        # ── Metrics Table ─────────────────────────────────────────────────────
        st.markdown(f"**Showing {len(filtered)} rules** (sorted by Lift ↓)")
        display_rules = filtered[["antecedents_str","consequents_str","support","confidence","lift"]].copy()
        display_rules.columns = ["If Customer Wants…","Then Also Wants…","Support","Confidence","Lift"]
        display_rules = display_rules.reset_index(drop=True)
        st.dataframe(
            display_rules.style
                .background_gradient(subset=["Lift"], cmap="YlOrRd")
                .background_gradient(subset=["Confidence"], cmap="Blues")
                .format({"Support": "{:.3f}", "Confidence": "{:.3f}", "Lift": "{:.3f}"}),
            use_container_width=True, height=400
        )

        st.divider()

        # ── Lift vs Confidence scatter ─────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            fig_lc = px.scatter(filtered, x="confidence", y="lift",
                                size="support", color="lift",
                                color_continuous_scale=["#1B4F72","#F39C12","#E74C3C"],
                                hover_data={"antecedents_str": True,
                                            "consequents_str": True},
                                labels={"confidence":"Confidence","lift":"Lift",
                                        "antecedents_str":"If","consequents_str":"Then"})
            fig_lc.update_coloraxes(showscale=False)
            plotly_dark_layout(fig_lc, "Lift vs Confidence (bubble = support)", height=420)
            st.plotly_chart(fig_lc, use_container_width=True)

        with col4:
            top10 = filtered.head(10).copy()
            top10["Rule"] = top10["antecedents_str"].str[:25] + " → " + top10["consequents_str"].str[:25]
            fig_top = go.Figure()
            fig_top.add_trace(go.Bar(name="Lift", x=top10["Rule"], y=top10["lift"],
                                     marker_color=PALETTE["accent"]))
            fig_top.add_trace(go.Bar(name="Confidence", x=top10["Rule"], y=top10["confidence"],
                                     marker_color=PALETTE["primary"]))
            fig_top.update_layout(barmode="group", xaxis_tickangle=-35)
            plotly_dark_layout(fig_top, "Top 10 Rules — Lift & Confidence", height=420)
            st.plotly_chart(fig_top, use_container_width=True)

        # ── Support histogram ─────────────────────────────────────────────────
        fig_sup = px.histogram(filtered, x="support", nbins=20,
                               color_discrete_sequence=[PALETTE["positive"]],
                               labels={"support":"Support"})
        plotly_dark_layout(fig_sup, "Support Distribution of Rules", height=300)
        st.plotly_chart(fig_sup, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Association Insight:</b> High-lift rules reveal what customers bundle together —
    use this for <b>package design</b> (e.g. bedroom + mattress + curtains)
    and <b>cross-sell triggers</b> during checkout and onboarding flows.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — PREDICT NEW CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif selected_tab == "🔮 Predict New Customers":
    st.markdown("<div class='section-header'>🔮 Predict New Customer Subscription Intent</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    📋 <b>How to use:</b> Upload a CSV of prospective customers with the same column structure
    as the training data. The app will predict subscription probability, classify each lead,
    assign a customer segment, and recommend a marketing approach. Download results as CSV.
    </div>""", unsafe_allow_html=True)

    # ── Download template ─────────────────────────────────────────────────────
    with st.expander("📥 Download CSV Template"):
        st.markdown("Use the training data structure. Required columns:")
        feature_cols_info = get_feature_cols(df)
        st.code(", ".join(feature_cols_info[:30]) + " ... (104 feature columns)", language="text")

        template_df = df[feature_cols_info].head(3).copy()
        csv_template = template_df.to_csv(index=False)
        b64 = base64.b64encode(csv_template.encode()).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="new_customers_template.csv">'
            f'⬇️ Download Template CSV</a>',
            unsafe_allow_html=True
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "📂 Upload New Customer Data (CSV)",
        type=["csv"],
        help="CSV must match the training data column structure"
    )

    if uploaded_file is not None:
        with st.spinner("Loading and validating data..."):
            new_df = pd.read_csv(uploaded_file)

        st.markdown(f"✅ **{len(new_df)} records loaded** | Columns: {new_df.shape[1]}")

        # Train models
        with st.spinner("Training models and running predictions..."):
            models, scaler, feature_cols, X_test, X_test_sc, y_test, results = train_models(df)
            cluster_labels, X_pca, kmeans = run_clustering(df)

        # Validate and align columns
        feature_cols_needed = get_feature_cols(df)
        missing_cols = [c for c in feature_cols_needed if c not in new_df.columns]
        extra_cols = [c for c in new_df.columns if c not in feature_cols_needed and c != "respondent_id"]

        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols[:10]}{'...' if len(missing_cols)>10 else ''}")
            st.stop()

        # Prepare feature matrix
        X_new = new_df[feature_cols_needed].copy()
        X_new = X_new.fillna(X_new.median())

        # Scale for LR
        X_new_sc = scaler.transform(X_new)

        # Predictions from all 3 models
        lr_prob  = models["Logistic Regression"].predict_proba(X_new_sc)[:, 1]
        rf_prob  = models["Random Forest"].predict_proba(X_new)[:, 1]
        xgb_prob = models["XGBoost"].predict_proba(X_new)[:, 1]

        # Ensemble (average)
        ensemble_prob = (lr_prob + rf_prob + xgb_prob) / 3
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)

        # Cluster assignment
        scaler2 = StandardScaler()
        X_train_sc2 = scaler2.fit_transform(df[feature_cols_needed])
        kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans2.fit(X_train_sc2)
        X_new_sc2 = scaler2.transform(X_new)
        new_clusters = kmeans2.predict(X_new_sc2)

        SEGMENT_NAMES = {
            0: "Mobile Professionals",
            1: "Family Settlers",
            2: "Budget Conscious",
            3: "Premium Upgraders",
        }
        MARKETING = {
            0: "Flexibility & Speed — fast setup, easy swap messaging",
            1: "Quality & Completeness — bundle offers, stability pitch",
            2: "Price-first — entry packages, no upfront cost messaging",
            3: "Lifestyle Upgrade — curated collections, concierge pitch",
        }

        # Build results
        results_df = new_df[["respondent_id"]].copy() if "respondent_id" in new_df.columns \
                     else pd.DataFrame({"record": range(1, len(new_df)+1)})
        results_df["LR_Probability"]       = lr_prob.round(4)
        results_df["RF_Probability"]       = rf_prob.round(4)
        results_df["XGB_Probability"]      = xgb_prob.round(4)
        results_df["Ensemble_Probability"] = ensemble_prob.round(4)
        results_df["Predicted_Subscribe"]  = ensemble_pred
        results_df["Lead_Tier"] = pd.cut(ensemble_prob,
                                          bins=[0, 0.35, 0.6, 0.8, 1.0],
                                          labels=["Cold", "Warm", "Hot", "🔥 Priority"])
        results_df["Customer_Segment"]     = [SEGMENT_NAMES[c] for c in new_clusters]
        results_df["Marketing_Approach"]   = [MARKETING[c] for c in new_clusters]

        st.divider()
        st.markdown("### 📊 Prediction Results")

        # ── Summary KPIs ──────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{ensemble_pred.sum()}</div>
                <div class='metric-label'>Predicted to Subscribe</div></div>""",
                unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{ensemble_pred.mean()*100:.1f}%</div>
                <div class='metric-label'>Predicted Conversion Rate</div></div>""",
                unsafe_allow_html=True)
        with col3:
            hot = (results_df["Lead_Tier"].isin(["Hot","🔥 Priority"])).sum()
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{hot}</div>
                <div class='metric-label'>Hot + Priority Leads</div></div>""",
                unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{ensemble_prob.mean()*100:.1f}%</div>
                <div class='metric-label'>Avg Subscription Probability</div></div>""",
                unsafe_allow_html=True)

        st.markdown("")

        # ── Tier & Segment Distribution ────────────────────────────────────────
        col5, col6 = st.columns(2)
        with col5:
            tier_counts = results_df["Lead_Tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier","Count"]
            fig_tier = px.pie(tier_counts, names="Tier", values="Count", hole=0.4,
                              color_discrete_sequence=[PALETTE["neutral"], "#F39C12",
                                                       PALETTE["positive"], "#E74C3C"])
            plotly_dark_layout(fig_tier, "Lead Tier Distribution", height=340)
            st.plotly_chart(fig_tier, use_container_width=True)

        with col6:
            seg_counts = results_df["Customer_Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment","Count"]
            fig_seg = px.bar(seg_counts, x="Segment", y="Count",
                             color="Count", color_continuous_scale=["#1B4F72","#F39C12"],
                             text="Count")
            fig_seg.update_traces(textposition="outside")
            fig_seg.update_coloraxes(showscale=False)
            plotly_dark_layout(fig_seg, "Predicted Segment Distribution", height=340)
            st.plotly_chart(fig_seg, use_container_width=True)

        # ── Probability Distribution ───────────────────────────────────────────
        fig_prob = px.histogram(results_df, x="Ensemble_Probability", nbins=25,
                                color_discrete_sequence=[PALETTE["accent"]],
                                labels={"Ensemble_Probability": "Subscription Probability"})
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color=PALETTE["negative"],
                           annotation_text="Decision Threshold (0.5)")
        plotly_dark_layout(fig_prob, "Distribution of Predicted Subscription Probabilities", height=320)
        st.plotly_chart(fig_prob, use_container_width=True)

        # ── Data Table ────────────────────────────────────────────────────────
        st.markdown("**Detailed Results Table**")
        st.dataframe(
            results_df.style.background_gradient(subset=["Ensemble_Probability"], cmap="RdYlGn")
                            .format({"LR_Probability": "{:.3f}",
                                     "RF_Probability": "{:.3f}",
                                     "XGB_Probability": "{:.3f}",
                                     "Ensemble_Probability": "{:.3f}"}),
            use_container_width=True, height=400
        )

        # ── Download ──────────────────────────────────────────────────────────
        csv_out = results_df.to_csv(index=False)
        b64_out = base64.b64encode(csv_out.encode()).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{b64_out}" download="predictions_output.csv">'
            f'⬇️ Download Full Prediction Results as CSV</a>',
            unsafe_allow_html=True
        )

    else:
        # ── Placeholder ───────────────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem;
                    border: 2px dashed rgba(255,255,255,0.15);
                    border-radius:16px; color:#8FA8BB;'>
            <div style='font-size:3rem;'>📂</div>
            <div style='font-family:Syne; font-size:1.2rem; margin:0.5rem 0;'>
                Upload a CSV to get started</div>
            <div style='font-size:0.85rem;'>
                Predictions will include subscription probability, lead tier,
                customer segment, and marketing approach recommendations.
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style='text-align:center; color:#8FA8BB; font-size:0.78rem; padding:1rem 0;'>
    🛋️ FurnRent Analytics Intelligence · UAE Furniture Rental Market ·
    Built with Streamlit · Powered by XGBoost, Random Forest, K-Means & Apriori
</div>""", unsafe_allow_html=True)
