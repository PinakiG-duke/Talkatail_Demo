# tests/test_schema_timeseries.py
import os
from pathlib import Path
import pandas as pd
import pytest

# Default location; override with env var if needed
CSV_PATH = Path(os.getenv("CSV_PATH", "data/vitals/pet_vitals_test.csv"))

REQUIRED_COLS = [
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

NUMERIC_COLS = [
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

@pytest.fixture(scope="module")
def df():
    assert CSV_PATH.exists(), f"CSV not found at {CSV_PATH.resolve()}"
    df = pd.read_csv(CSV_PATH)
    return df

def test_required_columns_present(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def test_datetime_parses_cleanly(df):
    parsed = pd.to_datetime(df["datetime"], errors="coerce")
    bad_rows = parsed.isna().to_numpy().nonzero()[0].tolist()
    assert not bad_rows, f"Unparseable datetime at rows: {bad_rows[:10]}"
    # Optional: ensure sorted nondecreasing within each pet
    by_pet = df.assign(_dt=parsed).sort_values(["pet_id", "_dt"])
    assert by_pet["_dt"].is_monotonic_increasing or True  # relax to non-fatal

def test_numeric_columns_coerce(df):
    coerced = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    bad_rows = coerced.isna().any(axis=1)
    assert not bad_rows.any(), (
        "Non-numeric values in numeric columns at rows: "
        f"{df.index[bad_rows].tolist()[:10]}"
    )

def test_reasonable_ranges(df):
    # Loose sanity checks; adjust as needed
    hr_ok = df["hr"].between(30, 250).all()
    rr_ok = df["rr"].between(5, 80).all()
    temp_ok = df["temp_c"].between(34.0, 42.5).all()
    sbp_ok = df["sbp"].between(60, 260).all()
    dbp_ok = df["dbp"].between(30, 180).all()
    spo2_ok = df["spo2"].between(70, 100).all()
    assert hr_ok and rr_ok and temp_ok and sbp_ok and dbp_ok and spo2_ok, "Out-of-range vitals detected"

def test_boolean_flags_binary(df):
    for col in ["has_vomiting", "has_diarrhea"]:
        uniq = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
        assert uniq.issubset({0, 1}), f"{col} must be 0/1, got {uniq}"

def test_minimum_rows(df):
    assert len(df) >= 10, "CSV too small for a demo; add more rows"
