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
    """Apply the logistic sigmoid element-wise.

        sigmoid(z) = 1 / (1 + exp(-z))

    Args:
        z: 1D float array of any length.

    Returns:
        np.ndarray of the same shape, every value in the range (0, 1).

    Requirement: the result must be finite for large-magnitude inputs.
        `sigmoid(np.array([-1000.0, 1000.0]))` must return finite numbers,
        never `nan`.
    """
    raise NotImplementedError
# ============================ END TODO (Task 1) ==============================


# ================ TODO (Task 2): softmax, written out by hand ================
def softmax_loop(z: list) -> list:
    """Softmax of a 1D input, in PURE PYTHON — no NumPy in this function.

        softmax(z)_i = exp(z_i) / sum_j exp(z_j)

    Use plain lists, `for` loops, and the `math` module only.

    Args:
        z: list of floats, length K >= 1.

    Returns:
        list of K floats that sums to 1.0.

    Requirement: the result must be finite for large-magnitude inputs.
        `softmax_loop([1000.0, 1001.0])` must not raise `OverflowError`.
    """
    raise NotImplementedError
# ============================ END TODO (Task 2) ==============================


# ================= TODO (Task 3): softmax, vectorised =======================
def softmax_np(z: np.ndarray) -> np.ndarray:
    """The same softmax as `softmax_loop`, vectorised with NumPy.

    No Python `for` loop over the elements of `z`.

    Args:
        z: 1D float array, length K >= 1.

    Returns:
        np.ndarray of K floats that sums to 1.0, matching `softmax_loop`
        to within floating-point tolerance.

    Requirement: the result must be finite for large-magnitude inputs.
        `softmax_np(np.array([1000.0, 1001.0]))` must not contain `nan`.
    """
    raise NotImplementedError
# ============================ END TODO (Task 3) ==============================


# ========================= TODO (Task 4): entropy ============================
def entropy(p: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution, in NATS (natural log).

        H(p) = - sum_k p_k * log(p_k)

    Args:
        p: 1D probability array of length K; non-negative and sums to 1.

    Returns:
        Python float, >= 0.

    Requirement: `p` may contain exact zeros. Follow the convention
        0 * log(0) = 0, so the result stays finite (never `nan`, never `inf`).
        A one-hot `p` must give exactly 0.0.
    """
    raise NotImplementedError
# ============================ END TODO (Task 4) ==============================


# ====================== TODO (Task 5): cross-entropy =========================
def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross-entropy of the prediction q relative to the target p, in nats.

        H(p, q) = - sum_k p_k * log(q_k)

    Args:
        p: 1D target distribution of length K (a one-hot vector is allowed).
        q: 1D predicted distribution of the same length; strictly positive
           entries are NOT guaranteed.

    Returns:
        Python float.

    Requirement: wherever p_k = 0 the term contributes 0, even if q_k = 0.
        Where p_k > 0 and q_k = 0 the true value is infinite; return a large
        finite number instead of `nan` or `inf`, so that training code built
        on this function does not break. Use `1e-12` as the smallest
        probability you take the logarithm of; Task 6 must use the same value.
        Do not modify the inputs in place.
    """
    raise NotImplementedError
# ============================ END TODO (Task 5) ==============================


# ====================== TODO (Task 6): KL divergence =========================
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence D_KL(p || q), in nats.

        D_KL(p || q) = sum_k p_k * log(p_k / q_k)

    Args:
        p: 1D target distribution of length K.
        q: 1D predicted distribution of the same length.

    Returns:
        Python float, >= 0, and exactly 0.0 when p and q are equal.

    Requirement: the same zero conventions as `cross_entropy`. Your result
        must satisfy the identity D_KL(p || q) = H(p, q) - H(p); a test
        checks it against your own `cross_entropy` and `entropy`.
    """
    raise NotImplementedError
# ============================ END TODO (Task 6) ==============================


# ================== TODO (Task 7): focal loss (from a paper) =================
def focal_loss(p: np.ndarray, q: np.ndarray, gamma: float = 2.0,
               alpha: np.ndarray = None) -> float:
    """Focal loss for one K-class example.

    Source: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
        https://arxiv.org/abs/1708.02002 — read Section 3.1 and Equations
        (4)-(5) only. The equation is not reproduced here; take it from
        the paper.

    The paper writes the binary case in terms of p_t, the predicted probability
    of the ground-truth class. Implement the K-class form: take the paper's
    per-class term, weight it by the target probability p_k, and sum over k.

    Args:
        p: 1D target distribution of length K (a one-hot vector is allowed).
        q: 1D predicted distribution of the same length.
        gamma: the paper's focusing parameter, >= 0.
        alpha: None for no class weighting, or a 1D array of length K holding
            the paper's alpha **for each class**. Note that the paper writes a
            single scalar alpha for the binary case; here it is one value per
            class, so `alpha[k]` weights class k.

    Returns:
        Python float.

    Requirement: with `gamma=0` and `alpha=None` this must return exactly the
        same value as `cross_entropy(p, q)` — a test checks that.
    """
    raise NotImplementedError
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
