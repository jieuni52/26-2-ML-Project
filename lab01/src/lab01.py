"""Lab 1 — Environment Setup & Pandas Data Handling (EDA).

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Run:      python src/lab01.py
Self-check: python -m pytest tests/ -q
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "20233542"   # TODO: your student id, e.g. "20261234"
STUDENT_NAME = "강지은"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cafe_sales.csv"


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY."""
    np.random.seed(seed)


def load_data(path: str | Path) -> pd.DataFrame:
    """DO NOT MODIFY. Loads the raw csv exactly as stored."""
    return pd.read_csv(path)


# ======================= TODO (Task 1): missing values =======================
def summarize_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    missing = missing.sort_values(ascending=False, kind="stable")
    return missing
# ============================ END TODO (Task 1) ==============================


# ======================== TODO (Task 2): cleaning ============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy(deep=True)
    clean = clean.drop_duplicates(keep="first").reset_index(drop=True)
    clean["unit_price"] = pd.to_numeric(clean["unit_price"].astype(str).str.replace(",", "", regex=False), errors="raise").astype(float)
    quantity_median = clean["quantity"].median()
    clean["quantity"] = pd.to_numeric(clean["quantity"].fillna(quantity_median).astype(int))
    missing_total = clean["total_price"].isna()
    clean.loc[missing_total, "total_price"] = clean.loc[missing_total, "unit_price"] * clean.loc[missing_total, "quantity"]
    customer_rating_mean = round(clean["customer_rating"].mean(), 2)
    clean["customer_rating"] = clean["customer_rating"].fillna(customer_rating_mean)
    return clean
# ============================ END TODO (Task 2) ==============================


# ======================= TODO (Task 3): outliers (IQR) =======================
def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> list:
    col = df[column]

    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outlier_mask = (col < lower_bound) | (col > upper_bound)
    outliers = sorted(int(idx) for idx in df[outlier_mask].index)
    
    return outliers
# ============================ END TODO (Task 3) ==============================


# ====================== TODO (Task 4): group statistics ======================
def compute_group_stats(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col)[value_col].agg(["count", "mean", "sum"])
    grouped["mean"] = grouped["mean"].round(2)
    grouped = grouped.sort_values(by="sum", ascending=False)
    return grouped    
# ============================ END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY (except nothing — really, do not modify).

    Runs the full EDA pipeline and writes results.json next to the repo root.
    """
    set_seed()
    t0 = time.time()
    raw = load_data(DATA_PATH)
    missing = summarize_missing(raw)
    clean = clean_data(raw)
    outliers = detect_outliers_iqr(clean, "quantity")
    stats = compute_group_stats(clean, "category", "total_price")
    results = {
        "lab": "lab01",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": {
            "n_rows_raw": int(len(raw)),
            "n_rows_clean": int(len(clean)),
            "n_duplicates_removed": int(len(raw) - len(clean)),
            "missing_total_raw": int(missing.sum()),
            "missing_total_clean": int(clean.isna().sum().sum()),
            "n_outliers_quantity": int(len(outliers)),
            "top_category_by_revenue": str(stats.index[0]),
            "mean_rating_clean": float(round(clean["customer_rating"].mean(), 3)),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
