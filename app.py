"""
Medical Appointment No-Show Intelligence Dashboard
====================================================
An interactive Streamlit application for exploring the Kaggle "Medical
Appointment No Shows" dataset (Brazil, 2016) and predicting the probability
that a patient will miss a scheduled appointment.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.features import DAY_ORDER, MODEL_FEATURES, clean_and_engineer, load_raw_data, time_based_split

# --------------------------------------------------------------------------
# Page configuration & theming
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Appointment No-Show Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#5B5FEF"
SUCCESS = "#18B26B"
DANGER = "#E5484D"
WARNING = "#F5A524"
INFO = "#3B82F6"
MUTED = "#6B7280"
BG_CARD = "#FFFFFF"

CUSTOM_CSS = f"""
<style>
    .main {{ background-color: #F4F5FA; }}
    #MainMenu, footer {{visibility: hidden;}}

    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid #ECECF4;
        border-radius: 14px;
        padding: 16px 18px 10px 18px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 600; }}

    h1, h2, h3 {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}

    .hero {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #8B5FE6 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }}
    .hero h1 {{ color: white; margin-bottom: 4px; font-size: 1.9rem; }}
    .hero p {{ color: rgba(255,255,255,0.9); margin: 0; font-size: 0.98rem; }}

    .insight-card {{
        background-color: {BG_CARD};
        border-left: 4px solid {PRIMARY};
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
        font-size: 0.92rem;
    }}
    section[data-testid="stSidebar"] {{ background-color: #FBFBFE; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHART_TEMPLATE = "plotly_white"
NO_SHOW_COLORS = {"Yes": DANGER, "No": SUCCESS}

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "appointments.csv"


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing appointment data...")
def get_data() -> pd.DataFrame:
    raw = load_raw_data(str(DATA_PATH))
    return clean_and_engineer(raw)


@st.cache_resource(show_spinner="Training the no-show prediction model...")
def get_model(df: pd.DataFrame):
    """Train the HistGradientBoosting model.

    The model is trained live (it only takes ~1-2 seconds on this dataset)
    and cached for the lifetime of the app process via st.cache_resource, so
    it only actually runs once per deployment restart. We deliberately do
    NOT ship a pre-pickled model artifact: pickle/joblib files are tied to
    the exact Python + scikit-learn version that created them, and hosts
    like Streamlit Community Cloud can run a different version than the one
    used to build the artifact, which breaks unpickling. Training fresh
    sidesteps that entire class of deployment bug.
    """
    train_df, test_df, cutoff = time_based_split(df, test_frac=0.2)
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.08, l2_regularization=1.0,
        random_state=42, class_weight="balanced",
    )
    model.fit(train_df[MODEL_FEATURES], train_df["no_show_flag"])
    return model, train_df, test_df, cutoff, "Hist Gradient Boosting"


@st.cache_data(show_spinner="Computing feature importance...")
def get_feature_importance(_model, _X_test, _y_test):
    """Permutation importance computed live against the held-out test set.

    Cheap enough (a few seconds) to run once per session and cache, and it
    avoids depending on any pre-computed artifact file.
    """
    result = permutation_importance(
        _model, _X_test, _y_test, n_repeats=3, random_state=42, n_jobs=-1
    )
    return pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance": result.importances_mean}
    ).sort_values("importance", ascending=False)


df = get_data()
model, train_df, test_df, cutoff_date, model_name = get_model(df)


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🩺 Filters")

date_min, date_max = df["appointment_day"].min().date(), df["appointment_day"].max().date()
date_range = st.sidebar.date_input(
    "Appointment date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)

gender_sel = st.sidebar.multiselect("Gender", options=["F", "M"], default=["F", "M"])

age_range = st.sidebar.slider("Age range", min_value=0, max_value=100, value=(0, 100))

age_group_options = list(df["age_group"].cat.categories)
age_group_sel = st.sidebar.multiselect("Age group", options=age_group_options, default=age_group_options)

neigh_options = sorted(df["neighbourhood"].unique())
neigh_sel = st.sidebar.multiselect("Neighbourhood (leave empty = all)", options=neigh_options, default=[])

st.sidebar.markdown("#### Health & behaviour")
sms_sel = st.sidebar.selectbox("SMS reminder received", options=["All", "Yes", "No"], index=0)
scholarship_sel = st.sidebar.selectbox("On Bolsa Família scholarship", options=["All", "Yes", "No"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {len(df):,} appointments · {df['patient_id'].nunique():,} unique patients\n\n"
    f"Model test window starts {cutoff_date.date()}"
)

# Apply filters
mask = (
    df["appointment_day"].dt.date.between(date_range[0], date_range[-1])
    & df["gender"].isin(gender_sel)
    & df["age"].between(age_range[0], age_range[1])
    & df["age_group"].isin(age_group_sel)
)
if neigh_sel:
    mask &= df["neighbourhood"].isin(neigh_sel)
if sms_sel != "All":
    mask &= df["sms_received"] == (1 if sms_sel == "Yes" else 0)
if scholarship_sel != "All":
    mask &= df["scholarship"] == (1 if scholarship_sel == "Yes" else 0)

fdf = df[mask].copy()

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🩺 Medical Appointment No-Show Intelligence</h1>
        <p>Exploratory analytics and a machine-learning model that predicts the probability
        a patient will miss their scheduled appointment — Brazilian public health system, 2016.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if fdf.empty:
    st.warning("No appointments match the current filters. Try widening your selection in the sidebar.")
    st.stop()

tab_overview, tab_demo, tab_schedule, tab_health, tab_model = st.tabs(
    ["📊 Overview", "👥 Demographics", "🗓️ Scheduling Patterns", "💊 Health & Behaviour", "🤖 Prediction Model"]
)

# --------------------------------------------------------------------------
# TAB 1 — Overview
# --------------------------------------------------------------------------
with tab_overview:
    total_appts = len(fdf)
    total_patients = fdf["patient_id"].nunique()
    no_shows = int(fdf["no_show_flag"].sum())
    no_show_rate = no_shows / total_appts * 100
    avg_wait = fdf["waiting_days"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Appointments", f"{total_appts:,}")
    c2.metric("Unique Patients", f"{total_patients:,}")
    c3.metric("No-Shows", f"{no_shows:,}")
    c4.metric("No-Show Rate", f"{no_show_rate:.1f}%")
    c5.metric("Avg. Waiting Days", f"{avg_wait:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        monthly = (
            fdf.assign(month=fdf["appointment_day"].dt.tz_localize(None).dt.to_period("M").astype(str))
            .groupby(["month", "no_show"], observed=True)
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            monthly, x="month", y="count", color="no_show", barmode="group",
            color_discrete_map=NO_SHOW_COLORS, template=CHART_TEMPLATE,
            labels={"month": "Month", "count": "Appointments", "no_show": "No-Show"},
            title="Appointments by Month",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        counts = fdf["no_show"].value_counts().reset_index()
        counts.columns = ["no_show", "count"]
        fig = px.pie(
            counts, names="no_show", values="count", hole=0.55,
            color="no_show", color_discrete_map=NO_SHOW_COLORS, template=CHART_TEMPLATE,
            title="Show vs. No-Show Split",
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔎 Key Insights")
    insight_cols = st.columns(3)
    top_wait_bucket = (
        fdf.groupby("waiting_bucket", observed=True)["no_show_flag"].mean().idxmax()
    )
    top_dow = fdf.groupby("appointment_dow_name", observed=True)["no_show_flag"].mean().idxmax()
    sms_effect = fdf.groupby("sms_received")["no_show_flag"].mean()
    sms_txt = (
        f"SMS reminders correlate with a **{'lower' if sms_effect.get(1,0) < sms_effect.get(0,0) else 'higher'}** "
        f"no-show rate ({sms_effect.get(1,0)*100:.1f}% with SMS vs {sms_effect.get(0,0)*100:.1f}% without)."
    )
    with insight_cols[0]:
        st.markdown(
            f'<div class="insight-card">⏳ Patients who wait <b>{top_wait_bucket}</b> before their '
            f'appointment have the highest no-show rate in the current selection.</div>',
            unsafe_allow_html=True,
        )
    with insight_cols[1]:
        st.markdown(
            f'<div class="insight-card">📅 <b>{top_dow}</b> has the highest no-show rate among weekdays '
            f'in the current selection.</div>',
            unsafe_allow_html=True,
        )
    with insight_cols[2]:
        st.markdown(f'<div class="insight-card">📩 {sms_txt}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# TAB 2 — Demographics
# --------------------------------------------------------------------------
with tab_demo:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            fdf, x="age", color="no_show", nbins=40, marginal="box",
            color_discrete_map=NO_SHOW_COLORS, template=CHART_TEMPLATE,
            title="Age Distribution", labels={"age": "Age", "no_show": "No-Show"},
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        age_grp = (
            fdf.groupby("age_group", observed=True)["no_show_flag"].mean().mul(100).reset_index()
        )
        fig = px.bar(
            age_grp, x="age_group", y="no_show_flag", template=CHART_TEMPLATE,
            color="no_show_flag", color_continuous_scale=["#18B26B", "#F5A524", "#E5484D"],
            labels={"age_group": "Age Group", "no_show_flag": "No-Show Rate (%)"},
            title="No-Show Rate by Age Group",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        gender_grp = fdf.groupby("gender")["no_show_flag"].agg(["mean", "count"]).reset_index()
        gender_grp["mean"] *= 100
        fig = px.bar(
            gender_grp, x="gender", y="mean", template=CHART_TEMPLATE, color="gender",
            color_discrete_map={"F": "#EC4899", "M": INFO},
            labels={"gender": "Gender", "mean": "No-Show Rate (%)"},
            title="No-Show Rate by Gender",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        top_neigh = (
            fdf.groupby("neighbourhood").size().sort_values(ascending=False).head(15).reset_index(name="count")
        )
        fig = px.bar(
            top_neigh.sort_values("count"), x="count", y="neighbourhood", orientation="h",
            template=CHART_TEMPLATE, color_discrete_sequence=[PRIMARY],
            labels={"count": "Appointments", "neighbourhood": ""},
            title="Top 15 Neighbourhoods by Appointment Volume",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 3 — Scheduling Patterns
# --------------------------------------------------------------------------
with tab_schedule:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            fdf, x="no_show", y="waiting_days", color="no_show", template=CHART_TEMPLATE,
            color_discrete_map=NO_SHOW_COLORS,
            labels={"waiting_days": "Days Between Scheduling and Appointment", "no_show": "No-Show"},
            title="Waiting Days by No-Show Status",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        wb_order = ["Same day", "1-3 days", "4-7 days", "8-14 days", "15-30 days", "30+ days"]
        wb = (
            fdf.groupby("waiting_bucket", observed=True)["no_show_flag"].mean().mul(100)
            .reindex(wb_order).reset_index()
        )
        fig = px.line(
            wb, x="waiting_bucket", y="no_show_flag", markers=True, template=CHART_TEMPLATE,
            labels={"waiting_bucket": "Waiting Time", "no_show_flag": "No-Show Rate (%)"},
            title="No-Show Rate vs. Waiting Time",
        )
        fig.update_traces(line_color=PRIMARY, marker=dict(size=9, color=DANGER))
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    heat = (
        fdf.groupby(["appointment_dow_name", "no_show"], observed=True).size().unstack(fill_value=0).reindex(DAY_ORDER)
    )
    heat_pct = heat.div(heat.sum(axis=1), axis=0) * 100
    fig = px.imshow(
        heat_pct[["No", "Yes"]].T, x=DAY_ORDER, y=["Show", "No-Show"], text_auto=".1f",
        color_continuous_scale="RdYlGn_r", template=CHART_TEMPLATE, aspect="auto",
        labels=dict(color="% of day's appts"),
        title="Show / No-Show Rate (%) by Day of Week",
    )
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        sms_grp = fdf.groupby("sms_received")["no_show_flag"].mean().mul(100).reset_index()
        sms_grp["sms_received"] = sms_grp["sms_received"].map({0: "No SMS", 1: "SMS Sent"})
        fig = px.bar(
            sms_grp, x="sms_received", y="no_show_flag", template=CHART_TEMPLATE,
            color="sms_received", color_discrete_sequence=[MUTED, PRIMARY],
            labels={"sms_received": "", "no_show_flag": "No-Show Rate (%)"},
            title="No-Show Rate: SMS Reminder Effect",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        hour_grp = fdf.groupby("scheduled_hour")["no_show_flag"].mean().mul(100).reset_index()
        fig = px.bar(
            hour_grp, x="scheduled_hour", y="no_show_flag", template=CHART_TEMPLATE,
            color_discrete_sequence=[INFO],
            labels={"scheduled_hour": "Hour Appointment Was Scheduled", "no_show_flag": "No-Show Rate (%)"},
            title="No-Show Rate by Booking Hour",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 4 — Health & Behaviour
# --------------------------------------------------------------------------
with tab_health:
    conditions = ["scholarship", "hypertension", "diabetes", "alcoholism", "handicap_flag"]
    labels_map = {
        "scholarship": "Bolsa Família Scholarship", "hypertension": "Hypertension",
        "diabetes": "Diabetes", "alcoholism": "Alcoholism", "handicap_flag": "Disability",
    }
    rows = []
    for c in conditions:
        rate_yes = fdf.loc[fdf[c] == 1, "no_show_flag"].mean() * 100 if (fdf[c] == 1).any() else np.nan
        rate_no = fdf.loc[fdf[c] == 0, "no_show_flag"].mean() * 100 if (fdf[c] == 0).any() else np.nan
        rows.append({"condition": labels_map[c], "Has condition": rate_yes, "No condition": rate_no})
    cond_df = pd.DataFrame(rows).melt(id_vars="condition", var_name="group", value_name="no_show_rate")
    fig = px.bar(
        cond_df, x="condition", y="no_show_rate", color="group", barmode="group",
        template=CHART_TEMPLATE, color_discrete_sequence=[DANGER, SUCCESS],
        labels={"condition": "", "no_show_rate": "No-Show Rate (%)", "group": ""},
        title="No-Show Rate by Health & Socioeconomic Factors",
    )
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        prior = fdf[fdf["prior_appointments"] > 0].copy()
        prior["prior_no_show_bucket"] = pd.cut(
            prior["prior_no_show_rate"], bins=[-0.01, 0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "0-25%", "25-50%", "50-75%", "75-100%"],
        )
        pns = prior.groupby("prior_no_show_bucket", observed=True)["no_show_flag"].mean().mul(100).reset_index()
        fig = px.bar(
            pns, x="prior_no_show_bucket", y="no_show_flag", template=CHART_TEMPLATE,
            color_discrete_sequence=[PRIMARY],
            labels={"prior_no_show_bucket": "Patient's Past No-Show Rate", "no_show_flag": "Current No-Show Rate (%)"},
            title="Past Behaviour Predicts Future Behaviour",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Sample of Filtered Data")
        display_cols = [
            "patient_id", "gender", "age", "neighbourhood", "waiting_days",
            "appointment_dow_name", "sms_received", "no_show",
        ]
        st.dataframe(
            fdf[display_cols].sample(min(200, len(fdf)), random_state=1).rename(
                columns={"appointment_dow_name": "day_of_week"}
            ),
            use_container_width=True, height=380,
        )
        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered data (CSV)", data=csv, file_name="filtered_appointments.csv")

# --------------------------------------------------------------------------
# TAB 5 — Prediction Model
# --------------------------------------------------------------------------
with tab_model:
    st.markdown(f"### Model: **{model_name}**")
    st.caption(
        "Trained on appointments scheduled before "
        f"**{cutoff_date.date()}** and evaluated on every appointment on/after that date "
        "(a chronological split, so the model is always tested on the 'future' relative to training — "
        "this avoids the optimistic bias a random split would give)."
    )

    X_test, y_test = test_df[MODEL_FEATURES], test_df["no_show_flag"]
    proba_test = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{roc_auc:.3f}")
    m2.metric("PR-AUC (Avg. Precision)", f"{pr_auc:.3f}")
    m3.metric("Training rows", f"{len(train_df):,}")
    m4.metric("Test rows", f"{len(test_df):,}")

    col1, col2 = st.columns(2)
    with col1:
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=MUTED, dash="dash")))
        fig.update_layout(
            title=f"ROC Curve (AUC = {roc_auc:.3f})", xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate", template=CHART_TEMPLATE, margin=dict(t=50, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        precision, recall, _ = precision_recall_curve(y_test, proba_test)
        baseline = y_test.mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Model", line=dict(color=DANGER, width=3)))
        fig.add_hline(y=baseline, line_dash="dash", line_color=MUTED, annotation_text="Baseline (random)")
        fig.update_layout(
            title=f"Precision-Recall Curve (AP = {pr_auc:.3f})", xaxis_title="Recall", yaxis_title="Precision",
            template=CHART_TEMPLATE, margin=dict(t=50, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns([1, 1.3])
    with col3:
        threshold = st.slider("Classification threshold", 0.05, 0.95, 0.5, 0.05)
        preds = (proba_test >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues", template=CHART_TEMPLATE,
            x=["Predicted: Show", "Predicted: No-Show"], y=["Actual: Show", "Actual: No-Show"],
            title="Confusion Matrix",
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fi = get_feature_importance(model, X_test, y_test)
        fi = fi.sort_values("importance", ascending=True).tail(12)
        fig = px.bar(
            fi, x="importance", y="feature", orientation="h", template=CHART_TEMPLATE,
            color_discrete_sequence=[PRIMARY],
            title="What Drives the Model's Predictions (Permutation Importance)",
            labels={"importance": "Importance", "feature": ""},
        )
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Try it: predict a single appointment")
    st.caption("Fill in a hypothetical patient/appointment and get the model's no-show probability.")

    with st.form("predict_form"):
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            p_age = st.number_input("Age", 0, 100, 35)
            p_gender = st.selectbox("Gender", ["F", "M"])
        with pc2:
            p_wait = st.number_input("Waiting days (scheduled → appointment)", 0, 200, 7)
            p_dow = st.selectbox("Appointment day of week", DAY_ORDER, index=0)
        with pc3:
            p_sms = st.selectbox("SMS reminder received", ["Yes", "No"])
            p_scholarship = st.selectbox("Bolsa Família scholarship", ["No", "Yes"])
        with pc4:
            p_chronic = st.multiselect("Chronic conditions", ["Hypertension", "Diabetes", "Alcoholism", "Disability"])
            p_prior_rate = st.slider("Patient's past no-show rate", 0.0, 1.0, 0.0, 0.05)

        submitted = st.form_submit_button("Predict no-show probability", use_container_width=True)

    if submitted:
        neigh_freq_default = float(df["neighbourhood_freq"].median())
        row = pd.DataFrame([{
            "age": p_age,
            "gender_code": 1 if p_gender == "M" else 0,
            "scholarship": 1 if p_scholarship == "Yes" else 0,
            "hypertension": 1 if "Hypertension" in p_chronic else 0,
            "diabetes": 1 if "Diabetes" in p_chronic else 0,
            "alcoholism": 1 if "Alcoholism" in p_chronic else 0,
            "handicap_flag": 1 if "Disability" in p_chronic else 0,
            "sms_received": 1 if p_sms == "Yes" else 0,
            "waiting_days": p_wait,
            "appointment_dow": DAY_ORDER.index(p_dow),
            "appointment_month_num": pd.Timestamp.now().month,
            "scheduled_hour": 9,
            "neighbourhood_freq": neigh_freq_default,
            "prior_appointments": 1 if p_prior_rate > 0 else 0,
            "prior_no_show_rate": p_prior_rate,
        }])[MODEL_FEATURES]

        prob = model.predict_proba(row)[0, 1]
        risk_label = "High" if prob >= 0.5 else ("Medium" if prob >= 0.3 else "Low")
        risk_color = DANGER if risk_label == "High" else (WARNING if risk_label == "Medium" else SUCCESS)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": f"No-Show Risk: {risk_label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 30], "color": "#E6F7EE"},
                    {"range": [30, 50], "color": "#FEF3D9"},
                    {"range": [50, 100], "color": "#FCE4E4"},
                ],
            },
        ))
        gauge.update_layout(height=320, margin=dict(t=50, l=20, r=20, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        st.info(
            "💡 This is an illustrative estimate from a statistical model trained on historical patterns — "
            "not a clinical or operational decision tool."
        )

st.markdown("---")
st.caption(
    "Data source: Kaggle — Medical Appointment No Shows (Brazil, 2016) · "
    "Built with Streamlit, scikit-learn & Plotly."
)
