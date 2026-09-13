"""Lab 0 — public tests. DO NOT MODIFY.

Run from the lab folder:  python -m pytest tests/ -q
"""
import ast
import inspect
import textwrap
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lab00 import (  # noqa: E402
    cross_entropy,
    entropy,
    focal_loss,
    kl_divergence,
    sigmoid,
    softmax_loop,
    softmax_np,
)

RNG = np.random.default_rng(42)


def random_distribution(k: int) -> np.ndarray:
    """A strictly positive distribution over k classes."""
    x = RNG.random(k) + 0.05
    return x / x.sum()


# ------------------------------- Task 1 -------------------------------------
def test_sigmoid_at_zero_is_half():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_is_bounded_and_increasing():
    z = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    s = sigmoid(z)
    assert np.all(s > 0.0) and np.all(s < 1.0)
    assert np.all(np.diff(s) > 0.0)


def test_sigmoid_symmetry():
    z = np.array([-3.0, -0.5, 0.7, 2.2])
    assert sigmoid(-z) == pytest.approx(1.0 - sigmoid(z))


def test_sigmoid_is_finite_for_large_inputs():
    s = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(s)), "sigmoid overflowed on large-magnitude input"


# ------------------------------- Task 2 -------------------------------------
def test_softmax_loop_returns_a_plain_list():
    out = softmax_loop([1.0, 2.0, 3.0])
    assert isinstance(out, list)
    assert all(isinstance(v, float) for v in out)


def test_softmax_loop_uses_no_numpy():
    tree = ast.parse(textwrap.dedent(inspect.getsource(softmax_loop)))
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            used |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            used.add(node.module.split(".")[0])
    assert not ({"np", "numpy"} & used), (
        "softmax_loop must be written in pure Python (no NumPy)."
    )


def test_softmax_loop_sums_to_one():
    out = softmax_loop([2.0, 1.0, 0.1, -1.5])
    assert sum(out) == pytest.approx(1.0)
    assert all(v > 0.0 for v in out)


def test_softmax_loop_is_finite_for_large_inputs():
    out = softmax_loop([1000.0, 1001.0])
    assert all(np.isfinite(v) for v in out)
    assert sum(out) == pytest.approx(1.0)


# ------------------------------- Task 3 -------------------------------------
def test_softmax_np_matches_softmax_loop():
    z = np.array([2.0, 1.0, 0.1, -1.5, 3.3])
    assert softmax_np(z) == pytest.approx(np.array(softmax_loop(z.tolist())))


def test_softmax_np_is_shift_invariant():
    z = np.array([0.4, -1.2, 2.0])
    assert softmax_np(z + 7.0) == pytest.approx(softmax_np(z))


def test_softmax_np_is_finite_for_large_inputs():
    out = softmax_np(np.array([1000.0, 1001.0]))
    assert np.all(np.isfinite(out)), "softmax_np overflowed on large input"
    assert out.sum() == pytest.approx(1.0)


# ------------------------------- Task 4 -------------------------------------
def test_entropy_of_uniform_is_log_k():
    k = 4
    assert entropy(np.full(k, 1.0 / k)) == pytest.approx(np.log(k))


def test_entropy_of_onehot_is_zero():
    assert entropy(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_entropy_is_a_nonnegative_float():
    h = entropy(random_distribution(6))
    assert isinstance(h, float)
    assert h >= 0.0


# ------------------------------- Task 5 -------------------------------------
def test_cross_entropy_with_itself_equals_entropy():
    p = random_distribution(5)
    assert cross_entropy(p, p) == pytest.approx(entropy(p))


def test_cross_entropy_is_at_least_entropy():
    p = random_distribution(5)
    q = random_distribution(5)
    assert cross_entropy(p, q) >= entropy(p) - 1e-12


def test_cross_entropy_handles_zero_probabilities():
    p = np.array([0.0, 1.0])
    q = np.array([1.0, 0.0])
    value = cross_entropy(p, q)
    assert np.isfinite(value), "cross_entropy must stay finite when q has a zero"
    assert value > 10.0


def test_cross_entropy_does_not_modify_its_inputs():
    p = np.array([0.0, 0.5, 0.5])
    q = np.array([0.2, 0.3, 0.5])
    cross_entropy(p, q)
    assert p.tolist() == [0.0, 0.5, 0.5]
    assert q.tolist() == [0.2, 0.3, 0.5]


# ------------------------------- Task 6 -------------------------------------
def test_kl_of_identical_distributions_is_zero():
    p = random_distribution(4)
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_nonnegative():
    for _ in range(5):
        p = random_distribution(4)
        q = random_distribution(4)
        assert kl_divergence(p, q) >= -1e-12


def test_kl_equals_cross_entropy_minus_entropy():
    p = random_distribution(5)
    q = random_distribution(5)
    assert kl_divergence(p, q) == pytest.approx(cross_entropy(p, q) - entropy(p))


def test_kl_is_asymmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


# ------------------------------- Task 7 -------------------------------------
def test_focal_loss_with_gamma_zero_is_cross_entropy():
    p = np.array([0.0, 1.0, 0.0, 0.0])
    q = softmax_np(np.array([2.0, 1.0, 0.1, -1.5]))
    assert focal_loss(p, q, gamma=0.0) == pytest.approx(cross_entropy(p, q))


def test_focal_loss_downweights_an_easy_example():
    p = np.array([0.0, 1.0])
    easy = np.array([0.02, 0.98])
    assert focal_loss(p, easy, gamma=2.0) < cross_entropy(p, easy)


def test_focal_loss_downweights_easy_more_than_hard():
    p = np.array([0.0, 1.0])
    easy = np.array([0.02, 0.98])
    hard = np.array([0.45, 0.55])
    easy_ratio = focal_loss(p, easy, gamma=2.0) / cross_entropy(p, easy)
    hard_ratio = focal_loss(p, hard, gamma=2.0) / cross_entropy(p, hard)
    assert easy_ratio < hard_ratio


def test_focal_loss_alpha_weights_the_target_class():
    p = np.array([0.0, 1.0])
    q = np.array([0.3, 0.7])
    base = focal_loss(p, q, gamma=2.0, alpha=None)
    weighted = focal_loss(p, q, gamma=2.0, alpha=np.array([0.75, 0.25]))
    assert weighted == pytest.approx(0.25 * base)
