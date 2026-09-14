"""Lab 1 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests with the same interfaces on different data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab01  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "cafe_sales.csv"


@pytest.fixture
def toy():
    return pd.DataFrame({
        "order_id": [1, 2, 3, 2, 4],
        "date": ["2026-09-01"] * 5,
        "item": ["americano", "latte", "scone", "latte", "americano"],
        "category": ["coffee", "coffee", "bakery", "coffee", "coffee"],
        "unit_price": [4500, "5,000", 4000, "5,000", 4500],
        "quantity": [1, 2, np.nan, 2, 100],
        "total_price": [4500, 10000, np.nan, 10000, np.nan],
        "customer_rating": [4.0, np.nan, 3.0, np.nan, 5.0],
        "takeout": ["yes", "no", "no", "no", "yes"],
    }).copy()
    # row index 3 duplicates row index 1 exactly


def test_summarize_missing_counts_and_order(toy):
    s = lab01.summarize_missing(toy)
    assert isinstance(s, pd.Series)
    assert s.to_dict() == {"quantity": 1, "total_price": 2, "customer_rating": 2}
    assert list(s.values) == sorted(s.values, reverse=True)


def test_clean_data_does_not_mutate_input(toy):
    snapshot = toy.copy(deep=True)
    lab01.clean_data(toy)
    pd.testing.assert_frame_equal(toy, snapshot)


def test_clean_data_toy(toy):
    c = lab01.clean_data(toy)
    assert len(c) == 4                                   # one duplicate dropped
    assert c.isna().sum().sum() == 0                     # no missing left
    assert c["unit_price"].dtype == float
    assert c.loc[c["order_id"] == 2, "unit_price"].iloc[0] == 5000.0
    assert np.issubdtype(c["quantity"].dtype, np.integer)
    # median quantity after dedup = median(1, 2, 100) = 2 -> filled scone row
    assert c.loc[c["item"] == "scone", "quantity"].iloc[0] == 2
    assert c.loc[c["item"] == "scone", "total_price"].iloc[0] == 8000.0
    # rating mean of (4.0, 3.0, 5.0) = 4.0
    assert c.loc[c["order_id"] == 2, "customer_rating"].iloc[0] == 4.0


def test_outliers_iqr_toy():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100, np.nan]})
    assert lab01.detect_outliers_iqr(df, "x") == [5]
    assert lab01.detect_outliers_iqr(df, "x", k=1000.0) == []


def test_group_stats_toy(toy):
    c = lab01.clean_data(toy)
    g = lab01.compute_group_stats(c, "category", "total_price")
    assert list(g.columns) == ["count", "mean", "sum"]
    assert list(g.index[:1]) == ["coffee"]               # coffee has the largest sum
    assert list(g["sum"].values) == sorted(g["sum"].values, reverse=True)


def test_end_to_end_on_real_data():
    raw = lab01.load_data(DATA)
    c = lab01.clean_data(raw)
    assert c.isna().sum().sum() == 0
    assert len(c) < len(raw)                             # duplicates existed
    outl = lab01.detect_outliers_iqr(c, "quantity")
    assert len(outl) >= 3                                # planted bulk orders
    g = lab01.compute_group_stats(c, "category", "total_price")
    assert set(g.index) <= {"coffee", "tea", "dessert", "bakery"}
