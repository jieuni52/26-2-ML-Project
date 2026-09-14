"""Lab 0 — Warm-up: version control and six machine-learning equations.

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — the tests call them directly.

Run:        python src/lab00.py
Self-check: python -m pytest tests/ -q
"""
import math
import time

import numpy as np


# ========================= TODO (Task 1): sigmoid ============================
def sigmoid(z: np.ndarray) -> np.ndarray:
    result = np.empty_like(z, dtype=float)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    result[~positive] = np.exp(z[~positive]) / (1 + np.exp(z[~positive]))
    return result
# ============================ END TODO (Task 1) ==============================


# ================ TODO (Task 2): softmax, written out by hand ================
def softmax_loop(z: list) -> list:
    max_z = max(z)
    exp = [math.exp(x - max_z) for x in z]
    sum_exp = sum(exp)
    return [x / sum_exp for x in exp]
# ============================ END TODO (Task 2) ==============================


# ================= TODO (Task 3): softmax, vectorised =======================
def softmax_np(z: np.ndarray) -> np.ndarray:
    max_z = np.max(z)
    exp = np.exp(z - max_z)
    sum_exp = np.sum(exp)
    return exp / sum_exp
# ============================ END TODO (Task 3) ==============================


# ========================= TODO (Task 4): entropy ============================
def entropy(p: np.ndarray) -> float:
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])))
# ============================ END TODO (Task 4) ==============================


# ====================== TODO (Task 5): cross-entropy =========================
def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    q_safe = np.maximum(q, 1e-12)
    return float(-np.sum(p * np.log(q_safe)))
# ============================ END TODO (Task 5) ==============================


# ====================== TODO (Task 6): KL divergence =========================
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(cross_entropy(p, q) - entropy(p))
# ============================ END TODO (Task 6) ==============================


# ================== TODO (Task 7): focal loss (from a paper) =================
def focal_loss(p: np.ndarray, q: np.ndarray, gamma: float = 2.0,
               alpha: np.ndarray = None) -> float:
    q_safe = np.maximum(q, 1e-12)
    factor = (1 - q_safe) ** gamma
    if alpha is None:
        return float(-np.sum(p * factor * np.log(q_safe)))
    else:
        return float(-np.sum(alpha * p * factor * np.log(q_safe)))
# ============================ END TODO (Task 7) ==============================


def main() -> None:
    """DO NOT MODIFY. Runs every function once and prints the numbers."""
    z = np.array([2.0, 1.0, 0.1, -1.5])
    p = np.array([0.0, 1.0, 0.0, 0.0])
    q = softmax_np(z)

    print("input z            :", np.round(z, 4).tolist())
    print("sigmoid(z)         :", np.round(sigmoid(z), 4).tolist())
    print("softmax_loop(z)    :", [round(v, 4) for v in softmax_loop(z.tolist())])
    print("softmax_np(z)      :", np.round(q, 4).tolist())
    print("target p (one-hot) :", p.tolist())
    print()
    print(f"entropy(q)             = {entropy(q):.6f}")
    print(f"entropy(p)             = {entropy(p):.6f}")
    print(f"cross_entropy(p, q)    = {cross_entropy(p, q):.6f}")
    print(f"kl_divergence(p, q)    = {kl_divergence(p, q):.6f}")
    print(f"H(p,q) - H(p)          = {cross_entropy(p, q) - entropy(p):.6f}")
    print(f"focal_loss(p, q, g=0)  = {focal_loss(p, q, gamma=0.0):.6f}")
    print(f"focal_loss(p, q, g=2)  = {focal_loss(p, q, gamma=2.0):.6f}")
    print()

    big = np.random.default_rng(42).normal(size=200000)
    big_list = big.tolist()
    t0 = time.perf_counter()
    softmax_loop(big_list)
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    softmax_np(big)
    t_np = time.perf_counter() - t0
    print(f"softmax over {len(big)} values")
    print(f"  pure Python : {t_loop * 1000:8.2f} ms")
    print(f"  NumPy       : {t_np * 1000:8.2f} ms")
    print(f"  speed-up    : {t_loop / max(t_np, 1e-9):8.1f}x")


if __name__ == "__main__":
    main()
