"""
Sanity tests for the permutation-test logic in tasks/evaluate.py.

These tests verify three properties:

1. Under H0 (same distribution) the p-value is not concentrated near 0.
2. Under H1 (different distributions) the test rejects.
3. Under H0, repeated p-values are roughly uniform on [0, 1].

We use a pure-numpy reference of the same statistic so the tests run
without requiring torch / a GPU. The tasks/evaluate.py implementation uses
the same kernel structure and the same permutation strategy.

Run with:  python tests/test_permutation.py
"""
from __future__ import annotations
import numpy as np


def gaussian_mmd2(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Unbiased Gaussian-MMD² (off-diagonal in K_xx, K_yy)."""
    def pairwise_d2(a, b):
        return (
            (a ** 2).sum(axis=1)[:, None]
            + (b ** 2).sum(axis=1)[None, :]
            - 2.0 * a @ b.T
        )

    K_xx = np.exp(-pairwise_d2(x, x) / (2.0 * sigma ** 2))
    K_yy = np.exp(-pairwise_d2(y, y) / (2.0 * sigma ** 2))
    K_xy = np.exp(-pairwise_d2(x, y) / (2.0 * sigma ** 2))

    n_x, n_y = x.shape[0], y.shape[0]
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)
    sum_xx = K_xx.sum() / (n_x * (n_x - 1))
    sum_yy = K_yy.sum() / (n_y * (n_y - 1))
    sum_xy = K_xy.mean()
    return sum_xx - 2.0 * sum_xy + sum_yy


def median_sigma(samples: np.ndarray) -> float:
    n = samples.shape[0]
    d2 = (
        (samples ** 2).sum(axis=1)[:, None]
        + (samples ** 2).sum(axis=1)[None, :]
        - 2.0 * samples @ samples.T
    )
    mask = ~np.eye(n, dtype=bool)
    return float(np.sqrt(max(np.median(d2[mask]), 1e-12)))


def perm_test(real, fake, num_permutations=500, seed=0):
    sigma = median_sigma(real)
    obs = gaussian_mmd2(real, fake, sigma)

    pooled = np.concatenate([real, fake], axis=0)
    n_real = real.shape[0]
    rng = np.random.default_rng(seed)

    null = np.empty(num_permutations)
    for i in range(num_permutations):
        idx = rng.permutation(pooled.shape[0])
        a = pooled[idx[:n_real]]
        b = pooled[idx[n_real:]]
        null[i] = gaussian_mmd2(a, b, sigma)

    p = (1.0 + np.sum(null >= obs)) / (1.0 + num_permutations)
    return obs, null, p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_same_distribution_does_not_reject():
    rng = np.random.default_rng(42)
    real = rng.standard_normal((60, 10))
    fake = rng.standard_normal((60, 10))
    obs, null, p = perm_test(real, fake, num_permutations=300, seed=7)
    print(f"[H0] obs={obs:+.4f}  null mean={null.mean():+.4f}  p={p:.3f}")
    assert p > 0.05, f"H0 wrongly rejected, p={p}"


def test_different_distributions_rejects():
    rng = np.random.default_rng(42)
    real = rng.standard_normal((60, 10))
    fake = rng.standard_normal((60, 10)) + 0.8        # shifted mean
    obs, null, p = perm_test(real, fake, num_permutations=300, seed=7)
    print(f"[H1] obs={obs:+.4f}  null mean={null.mean():+.4f}  p={p:.3f}")
    assert obs > 0, "MMD² should be positive when distributions differ"
    assert p < 0.05, f"H1 wrongly accepted, p={p}"


def test_pvalue_uniform_under_null():
    """Under H0, repeated p-values should be roughly uniform on [0, 1]."""
    from scipy.stats import kstest
    rng = np.random.default_rng(0)
    n_trials = 80
    pvals = []
    for t in range(n_trials):
        real = rng.standard_normal((40, 5))
        fake = rng.standard_normal((40, 5))
        _, _, p = perm_test(real, fake, num_permutations=200, seed=t)
        pvals.append(p)
    pvals = np.array(pvals)
    stat, ks_p = kstest(pvals, "uniform")
    print(f"[uniformity] n_trials={n_trials}  KS={stat:.3f}  KS-p={ks_p:.3f}  "
          f"mean(p)={pvals.mean():.3f}")
    assert ks_p > 0.01, f"p-values not uniform under H0 (KS p={ks_p})"
    assert 0.35 < pvals.mean() < 0.65, f"mean p-value off: {pvals.mean()}"


if __name__ == "__main__":
    test_same_distribution_does_not_reject()
    test_different_distributions_rejects()
    test_pvalue_uniform_under_null()
    print("\n✓ All permutation-test sanity checks passed.")
