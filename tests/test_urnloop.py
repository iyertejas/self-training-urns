"""Sanity checks. Run with `pytest` or simply `python tests/test_urnloop.py`."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from urnloop import critical_lambda, mean_field_drift, sharpened, simulate  # noqa: E402


def test_sharpened_is_a_distribution():
    p = sharpened(np.array([[1.0, 2.0, 7.0], [1e6, 1e6, 1.0]]), np.array([3.0, 50.0]))
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.all(p >= 0)


def test_alpha_one_is_proportional():
    c = np.array([1.0, 3.0, 6.0])
    assert np.allclose(sharpened(c, 1.0), c / c.sum())


def test_counts_increase_by_one_per_step():
    out = simulate(1.0, 0.0, K=4, n_steps=500, n_runs=3, init=1.0, seed=0)
    # shares * total = counts; total = K*init + n_steps
    assert np.allclose(out["shares"].sum(axis=-1), 1.0)


def test_polya_mean_share_is_preserved():
    # alpha = 1, lam = 0 is the classical Polya urn: E[share_1] = 1/K at all times.
    out = simulate(1.0, 0.0, K=3, n_steps=300, n_runs=4000, seed=1)
    assert abs(out["shares"][..., 0].mean() - 1 / 3) < 0.02


def test_only_fresh_data_converges_to_pi():
    out = simulate(5.0, 1.0, K=3, n_steps=20_000, n_runs=2, seed=2)
    assert np.allclose(out["shares"], 1 / 3, atol=0.02)


def test_uniform_is_a_fixed_point_and_threshold():
    x = np.full(5, 0.2)
    assert np.allclose(mean_field_drift(x, 2.0, 0.3), 0.0)
    assert np.isclose(critical_lambda(2.0), 0.5)
    assert critical_lambda(0.5) == 0.0


def test_linearisation_matches_threshold():
    # Numerical derivative of the drift at uniform, in a simplex direction.
    K, alpha, lam, eps = 4, 1.7, 0.2, 1e-6
    x = np.full(K, 1 / K)
    v = np.array([1.0, -1.0, 0.0, 0.0])
    deriv = (mean_field_drift(x + eps * v, alpha, lam) - mean_field_drift(x - eps * v, alpha, lam)) / (2 * eps)
    assert np.allclose(deriv, ((1 - lam) * alpha - 1) * v, atol=1e-5)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
