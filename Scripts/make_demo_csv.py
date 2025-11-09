"""
make_demo_vitals_csv.py
---------------------------------
Create a small demo CSV with random but realistic pet vital records.

Usage:
    python scripts/make_demo_vitals_csv.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# --- Config ---
N_RECORDS = 48                # number of rows to generate
N_PETS = 2                    # how many pet IDs
OUT_PATH = Path("data/vitals/pet_vitals_test.csv")
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Generate synthetic timeseries ---
rows = []
base_time = datetime.now().replace(minute=0, second=0, microsecond=0)

for pet_id in range(1, N_PETS + 1):
    for i in range(N_RECORDS):
        dt = base_time - timedelta(hours=(N_RECORDS - i))
        age = np.random.randint(1, 12)
        weight_kg = np.clip(np.random.normal(18, 4), 4, 45)
        hr = np.clip(np.random.normal(90, 10), 50, 180)
        rr = np.clip(np.random.normal(22, 3), 8, 50)
        temp_c = np.clip(np.random.normal(38.6, 0.4), 36.5, 40.5)
        sbp = np.clip(np.random.normal(120, 10), 85, 180)
        dbp = np.clip(np.random.normal(75, 6), 50, 120)
        spo2 = np.clip(np.random.normal(97, 1.5), 85, 100)
        activity_level = np.random.randint(0, 4)
        appetite_score = np.random.randint(0, 6)
        has_vomiting = np.random.randint(0, 2)
        has_diarrhea = np.random.randint(0, 2)

        rows.append(
            {
                "pet_id": pet_id,
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "age": age,
                "weight_kg": round(weight_kg, 1),
                "hr": int(hr),
                "rr": int(rr),
                "temp_c": round(temp_c, 1),
                "sbp": int(sbp),
                "dbp": int(dbp),
                "spo2": int(spo2),
                "activity_level": activity_level,
                "appetite_score": appetite_score,
                "has_vomiting": has_vomiting,
                "has_diarrhea": has_diarrhea,
            }
        )

df = pd.DataFrame(rows)
df = df.sort_values(["pet_id", "datetime"])

# --- Save ---
df.to_csv(OUT_PATH, index=False)
print(f"✅ Demo CSV written to: {OUT_PATH.resolve()}")
print(df.head())
