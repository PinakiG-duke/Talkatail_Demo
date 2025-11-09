# app_health_timeseries_plotly.py
# Train on synthetic if needed -> read CSV with pet_id, datetime -> write predictions back -> Streamlit charts

import os, time, pathlib
import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import plotly.express as px

# ----- CONFIG -----
CSV_PATH = r"D:\Ops\Hackathon Models\pet_vitals_test.csv"  # change if needed
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "health_lr.joblib"

REQUIRED = [
    "pet_id",
    "datetime",
    "age",
    "weight_kg",
    "hr",
    "rr",
    "temp_c",
    "sbp",
    "dbp",
    "spo2",
    "activity_level",
    "appetite_score",
    "has_vomiting",
    "has_diarrhea",
]
FEATURES = [
    "age",
    "weight_kg",
    "hr",
    "rr",
    "temp_c",
    "sbp",
    "dbp",
    "spo2",
    "activity_level",
    "appetite_score",
    "has_vomiting",
    "has_diarrhea",
]


# ----- MODEL -----
def synth(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(1, 16, n),
            "weight_kg": np.clip(rng.normal(18, 8, n), 1.5, 60),
            "hr": np.clip(rng.normal(90, 15, n), 40, 200),
            "rr": np.clip(rng.normal(22, 5, n), 8, 60),
            "temp_c": np.clip(rng.normal(38.5, 0.5, n), 36, 41.5),
            "sbp": np.clip(rng.normal(120, 12, n), 80, 200),
            "dbp": np.clip(rng.normal(75, 8, n), 40, 140),
            "spo2": np.clip(rng.normal(97, 2, n), 75, 100),
            "activity_level": rng.integers(0, 4, n),
            "appetite_score": rng.integers(0, 6, n),
            "has_vomiting": rng.integers(0, 2, n),
            "has_diarrhea": rng.integers(0, 2, n),
        }
    )
    score = (
        0.9 * (df.spo2 - 92)
        - 0.5 * np.maximum(0, np.abs(df.hr - 95) - 20)
        - 0.6 * np.maximum(0, np.abs(df.rr - 22) - 8)
        - 4.5 * np.maximum(0, np.abs(df.temp_c - 38.6) - 0.7)
        - 0.03 * np.maximum(0, df.sbp - 160)
        - 0.04 * np.maximum(0, 90 - df.dbp)
        + 1.5 * df.activity_level
        + 1.2 * df.appetite_score
        - 3.0 * df.has_vomiting
        - 2.5 * df.has_diarrhea
    )
    df["healthy"] = (score + rng.normal(0, 2.0, len(df)) > 2.5).astype(int)
    return df


@st.cache_resource(show_spinner=True)
def train_or_load():
    if MODEL_PATH.exists():
        return load(MODEL_PATH)
    df = synth()
    X, y = df[FEATURES], df["healthy"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipe.fit(Xtr, ytr)
    proba_te = pipe.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba_te)
    dump({"model": pipe, "auc": float(auc), "threshold": 0.5}, MODEL_PATH)
    return load(MODEL_PATH)


# ----- I/O -----
def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isna().any():
        bad = df.index[df["datetime"].isna()].tolist()[:10]
        raise ValueError(f"Bad datetime at rows: {bad}")
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if df[FEATURES].isna().any().any():
        bad = df.index[df[FEATURES].isna().any(axis=1)].tolist()[:10]
        raise ValueError(f"Non-numeric in features at rows: {bad}")
    return df


def backup_and_overwrite(csv_path: str, df_out: pd.DataFrame) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = f"{csv_path}.bak_{ts}"
    with open(csv_path, "rb") as fr, open(bak, "wb") as fw:
        fw.write(fr.read())
    df_out.to_csv(csv_path, index=False)
    return bak


# ----- UI -----
st.set_page_config(page_title="Pet Health Time-series", page_icon="🐾", layout="wide")
st.title("🐾 Pet Health: Predictions + Time-series")

if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found: {CSV_PATH}")
    st.stop()

model = train_or_load()

try:
    df = pd.read_csv(CSV_PATH)
    df = ensure_schema(df)
    proba = model["model"].predict_proba(df[FEATURES])[:, 1]
    label = (proba >= model["threshold"]).astype(int)
    df["health_proba"] = proba
    df["health_label"] = np.where(label == 1, "Healthy", "Not healthy")
    backup = backup_and_overwrite(CSV_PATH, df)
    st.caption(f"Predictions updated. Backup: {backup} | AUC={model['auc']:.3f}")
except Exception as e:
    st.error(f"Error processing CSV: {e}")
    st.stop()

# Filters
pets = sorted(df["pet_id"].astype(str).unique().tolist())
sel_pet = st.sidebar.selectbox("Pet", pets, index=0)
pet_df = df[df["pet_id"].astype(str) == sel_pet].copy().sort_values("datetime")

min_d, max_d = pet_df["datetime"].min().date(), pet_df["datetime"].max().date()
start_d, end_d = st.sidebar.date_input(
    "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
)
if isinstance(start_d, tuple):  # old Streamlit guard
    start_d, end_d = start_d
mask = (pet_df["datetime"].dt.date >= start_d) & (pet_df["datetime"].dt.date <= end_d)
pet_df = pet_df.loc[mask]

if pet_df.empty:
    st.warning("No data in selected range.")
    st.stop()

latest = pet_df.iloc[-1]

# Tiles
left, right = st.columns([1.3, 1.0], gap="large")
with left:
    st.subheader(f"Health Trends & Insights — Pet {sel_pet}")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    prev = pet_df.iloc[-2] if len(pet_df) >= 2 else latest
    delta_w = latest["weight_kg"] - prev["weight_kg"]
    delta_hr = latest["hr"] - prev["hr"]
    delta_steps = 200 * (latest["activity_level"] - prev["activity_level"])
    delta_temp = latest["temp_c"] - prev["temp_c"]

    with c1:
        st.subheader("Weight")
        st.metric("kg", f"{latest['weight_kg']:.1f}", f"{delta_w:+.1f} kg")
        st.caption("Recent weight change")
    with c2:
        st.subheader("Heart Rate")
        st.metric("bpm avg", f"{int(latest['hr'])}", f"{delta_hr:+.0f} bpm")
        st.caption("Short-term HR trend")
    with c3:
        st.subheader("Activity Level")
        steps = int(3500 + 200 * latest["activity_level"])
        st.metric("steps/day", f"{steps:,}", f"{delta_steps:+.0f} steps")
        st.caption("Daily activity")
    with c4:
        st.subheader("Sleep Quality")
        sleep_avg = 7.5 + 0.2 * latest["appetite_score"]
        st.metric("hrs avg", f"{sleep_avg:.1f}", "+0.5 hrs")
        st.caption("Heuristic proxy")
    with c5:
        st.subheader("Blood Pressure")
        st.metric("mmHg", f"{int(latest['sbp'])}/{int(latest['dbp'])}", "Stable")
        st.caption("BP status")
    with c6:
        st.subheader("Temperature")
        st.metric("°C", f"{latest['temp_c']:.1f}", f"{delta_temp:+.1f}°C")
        st.caption("Monitor if rising")

with right:
    score = int(round(latest["health_proba"] * 100))
    fig = px.pie(names=["Score", "Remaining"], values=[score, 100 - score], hole=0.6)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_traces(textinfo="none")
    fig.add_annotation(
        x=0.5, y=0.5, text=f"{score}<br>/ 100", showarrow=False, font_size=18
    )
    st.subheader("Overall Health Score")
    st.plotly_chart(fig, use_container_width=True)

# -------- Time-series (edited to your spec) --------
st.markdown("### Vitals over time — HR, RR, Temp, SBP, DBP")
pet_df = pet_df.sort_values("datetime")

# Single graph with the five requested vitals
vitals = pet_df.set_index("datetime")[["hr", "rr", "temp_c", "sbp", "dbp"]].rename(
    columns={
        "hr": "HR (bpm)",
        "rr": "RR (breaths/min)",
        "temp_c": "Temp (°C)",
        "sbp": "SBP (mmHg)",
        "dbp": "DBP (mmHg)",
    }
)
st.line_chart(vitals)

# Activity level
st.markdown("### Activity Level")
st.line_chart(
    pet_df.set_index("datetime")[["activity_level"]].rename(
        columns={"activity_level": "Activity Level"}
    )
)

# Appetite score
st.markdown("### Appetite Score")
st.line_chart(
    pet_df.set_index("datetime")[["appetite_score"]].rename(
        columns={"appetite_score": "Appetite Score"}
    )
)

# Overall health score over time (keep as before)
st.markdown("### Overall Health Score over time")
st.line_chart(
    pet_df.set_index("datetime")[["health_proba"]].rename(
        columns={"health_proba": "P(Healthy)"}
    )
)

st.caption(f"{len(pet_df)} records from {start_d} to {end_d} for pet {sel_pet}.")
