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
STUDENT_ID = "00000000"   # TODO: your student id, e.g. "20261234"
STUDENT_NAME = "None"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

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
    """Return the number of missing (NaN) values per column.

    Returns:
        pd.Series indexed by column name, integer counts, sorted in
        DESCENDING order of count (ties: keep pandas' stable order).
        Include only columns that have at least one missing value.
    """
    raise NotImplementedError
# ============================ END TODO (Task 1) ==============================


# ======================== TODO (Task 2): cleaning ============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned COPY of df (do not mutate the input). Steps, in order:

    1. Drop exact duplicate rows (keep the first occurrence), reset the index.
    2. Convert `unit_price` to float. Some values are strings with a
       thousands separator, e.g. "4,500" -> 4500.0.
    3. Fill missing `quantity` with the MEDIAN of the non-missing quantities
       (computed AFTER step 1), then cast `quantity` to int.
    4. Recompute `total_price` = unit_price * quantity for rows where
       `total_price` is missing; leave existing values untouched.
    5. Fill missing `customer_rating` with the column MEAN (computed after
       step 1) rounded to 2 decimals — i.e. the FILL VALUE is `round(mean, 2)`;
       do not round the existing ratings or the whole column.

    The returned frame must contain no missing values.
    """
    raise NotImplementedError
# ============================ END TODO (Task 2) ==============================


# ======================= TODO (Task 3): outliers (IQR) =======================
def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> list:
    """Return the sorted list of index labels whose `column` value is an
    outlier under the IQR rule:

        value < Q1 - k*IQR   or   value > Q3 + k*IQR,
        where IQR = Q3 - Q1 (Q1/Q3 = 25th/75th percentiles, pandas default).

    NaN values are never outliers. Return a plain Python list of ints.
    """
    raise NotImplementedError
# ============================ END TODO (Task 3) ==============================


# ====================== TODO (Task 4): group statistics ======================
def compute_group_stats(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Group df by `group_col` and aggregate `value_col`.

    Returns:
        DataFrame indexed by the group keys with exactly three columns
        ["count", "mean", "sum"] (count of non-missing values, mean rounded
        to 2 decimals, sum), sorted by "sum" in DESCENDING order.
    """
    raise NotImplementedError
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
