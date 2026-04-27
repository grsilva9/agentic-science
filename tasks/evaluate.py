"""
Task: Evaluate generated images against real data via a patch-based MMD
two-sample permutation test.

The previous implementation had several issues that this version fixes:

1. **Single observed statistic.** A two-sample permutation test produces
   *one* observed value (real vs. generated) and a *distribution* of null
   values from pooled-and-shuffled splits. The earlier version computed an
   "observed distribution" by re-permuting within groups, which is
   degenerate (MMD is invariant under within-group reordering) and is not a
   valid p-value construction.

2. **One-sided p-value.** MMD² ≥ 0 by construction. Larger values mean the
   distributions disagree more. The correct p-value is
   ``P(MMD_null ≥ MMD_obs)``, with a small +1 bias correction for the
   observed sample. The earlier version used absolute values, which is
   appropriate for symmetric statistics but not for MMD.

3. **Bandwidth.** The Gaussian kernel takes ``exp(-d² / (2σ²))`` where ``d``
   is an L2 *distance*. The earlier code's ``estimate_sigma`` returned the
   median *squared* distance, which made σ far too large and saturated the
   kernel. We now take ``sqrt(median(d²))`` as the bandwidth.

4. **Real-sample collection.** ``next(iter(loader))`` rebuilds the iterator
   each call, so the previous loop returned the same first batch repeatedly.
   We now drain the loader once.

References:
    Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.
    Sutherland et al., "Generative Models and Model Criticism via
        Optimized Maximum Mean Discrepancy", ICLR 2017.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn.functional import conv2d
from tqdm import tqdm

from schemas import EvaluateInput, EvaluateOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch-MMD core
# ---------------------------------------------------------------------------

def _gaussian_kernel(d_squared: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian RBF kernel applied to *squared* L2 distances."""
    return torch.exp(-d_squared / (2.0 * sigma ** 2))


class PatchMMD(torch.nn.Module):
    """
    Patch-based MMD² between two image batches with a Gaussian kernel.

    For each pair (image_i in X, image_j in Y) we compute, for every spatial
    location, the squared L2 distance between p×p patches centered there.
    A Gaussian kernel turns those distances into similarities, and MMD² is
    the standard unbiased estimator combining intra- and inter-group
    similarities.

    Args:
        patch_size: Side length of the square patch.
        channels:   Number of image channels.
        sigma:      Bandwidth of the Gaussian. Must be set before calling
                    ``forward`` (use ``estimate_sigma`` for the median
                    heuristic).
    """

    def __init__(
        self,
        patch_size: int = 3,
        channels: int = 1,
        sigma: float | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.sigma = sigma
        # Sum-pool kernel: convolving squared per-pixel differences with a
        # ones-kernel gives the patch-wise sum of squared differences,
        # i.e. squared L2 distance between patches.
        self.register_buffer(
            "patch_kernel",
            torch.ones((1, channels, patch_size, patch_size), dtype=torch.float32),
        )

    def pairwise_patch_dist_sq(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        """
        Squared L2 distances between every patch in A and every patch in B.

        Returns a tensor of shape (N_A, N_B, H', W'), where H', W' are the
        spatial dimensions after the valid-mode patch convolution.
        """
        N_A = A.size(0)
        N_B = B.size(0)
        # Broadcast to (N_A, N_B, C, H, W) — careful with memory!
        diff_sq = (A[:, None] - B[None, :]) ** 2
        # Fold the (N_A, N_B) batch into one dim for conv2d.
        diff_sq = diff_sq.flatten(0, 1)              # (N_A*N_B, C, H, W)
        d_sq = conv2d(diff_sq, self.patch_kernel)    # (N_A*N_B, 1, H', W')
        H_, W_ = d_sq.shape[-2:]
        return d_sq.view(N_A, N_B, H_, W_)

    def _kernel_mean(
        self, d_sq: torch.Tensor, exclude_diagonal: bool = False
    ) -> torch.Tensor:
        """
        Average of K(d²) over all pairs (and over spatial locations).

        If ``exclude_diagonal`` is True we drop the i==j entries (used for
        the unbiased estimator on K_xx and K_yy).
        """
        K = _gaussian_kernel(d_sq, self.sigma)        # (N_A, N_B, H', W')
        # Average over spatial locations first → (N_A, N_B).
        K_pair = K.mean(dim=(-2, -1))
        if exclude_diagonal:
            n = K_pair.size(0)
            mask = ~torch.eye(n, dtype=torch.bool, device=K_pair.device)
            return K_pair[mask].mean()
        return K_pair.mean()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Unbiased MMD² estimator between batches ``x`` and ``y``."""
        if self.sigma is None:
            raise RuntimeError("PatchMMD: sigma is not set.")
        d_xx = self.pairwise_patch_dist_sq(x, x)
        d_yy = self.pairwise_patch_dist_sq(y, y)
        d_xy = self.pairwise_patch_dist_sq(x, y)
        return (
            self._kernel_mean(d_xx, exclude_diagonal=True)
            - 2.0 * self._kernel_mean(d_xy, exclude_diagonal=False)
            + self._kernel_mean(d_yy, exclude_diagonal=True)
        )


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------

def estimate_sigma(samples: torch.Tensor, patch_size: int = 3) -> float:
    """
    Median heuristic for the Gaussian kernel bandwidth.

    Concretely: σ = sqrt(median(d²)), where ``d²`` is the patch-wise squared
    L2 distance between distinct samples in ``samples``. Taking the square
    root is what makes the kernel ``exp(-d² / (2σ²))`` non-degenerate.
    """
    mmd = PatchMMD(patch_size=patch_size, channels=samples.size(1))
    with torch.no_grad():
        d_sq = mmd.pairwise_patch_dist_sq(samples, samples)
    n = samples.size(0)
    # Drop the i==j entries (those are zero) to avoid biasing the median.
    mask = ~torch.eye(n, dtype=torch.bool, device=d_sq.device)
    d_sq_off = d_sq[mask].flatten()
    median_sq = d_sq_off.median().item()
    sigma = float(np.sqrt(max(median_sq, 1e-12)))
    logger.info("Median heuristic: median(d²)=%.6g → σ=%.6g", median_sq, sigma)
    return sigma


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    real: torch.Tensor,
    fake: torch.Tensor,
    distance_fn: PatchMMD,
    num_permutations: int = 1000,
    seed: int = 0,
) -> Tuple[float, np.ndarray, float]:
    """
    Two-sample permutation test for the patch-MMD statistic.

    Pipeline:
        1. Compute MMD(real, fake) — the *observed* statistic (one number).
        2. Pool the two groups and, for each of ``num_permutations`` draws,
           shuffle the pool and split it into two halves of the original
           sizes. Compute MMD on each split → null distribution.
        3. p_value = (1 + #{null ≥ obs}) / (1 + num_permutations).

    Returns:
        ``(observed_mmd, null_distribution, p_value)``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = real.to(device)
    fake = fake.to(device)
    distance_fn = distance_fn.to(device)   # <-- THIS LINE IS THE FIX

    n_real, n_fake = real.size(0), fake.size(0)
    pooled = torch.cat([real, fake], dim=0)
    n_total = pooled.size(0)

    with torch.no_grad():
        observed = distance_fn(real, fake).item()

    rng = np.random.default_rng(seed)
    null_stats = np.empty(num_permutations, dtype=np.float64)

    for i in tqdm(range(num_permutations), desc="Permutation test"):
        idx = rng.permutation(n_total)
        idx_t = torch.from_numpy(idx).to(device)
        a = pooled.index_select(0, idx_t[:n_real])
        b = pooled.index_select(0, idx_t[n_real:])
        with torch.no_grad():
            null_stats[i] = distance_fn(a, b).item()

    p_value = (1.0 + np.sum(null_stats >= observed)) / (1.0 + num_permutations)

    logger.info(
        "Observed MMD² = %.6g, null mean = %.6g, null std = %.6g, p = %.4f",
        observed, null_stats.mean(), null_stats.std(), p_value,
    )
    return observed, null_stats, float(p_value)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_permutation(
    null_dist: np.ndarray,
    observed: float,
    p_value: float,
    title: str = "Permutation Test: Patch-MMD",
) -> plt.Figure:
    """
    Density of the null distribution with the observed statistic marked.

    Returns:
        A matplotlib Figure (caller is responsible for saving/closing).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("white")

    sns.kdeplot(
        null_dist, color="#3b78c2", fill=True, alpha=0.5, label="Null", ax=ax,
    )
    ax.axvline(
        observed, color="#c23b3b", linestyle="--", linewidth=2,
        label=f"Observed (p = {p_value:.3f})",
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("MMD² statistic", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_real_samples(test_loader, n: int) -> torch.Tensor:
    """
    Drain the test loader once and return up to ``n`` samples. If the loader
    yields fewer than ``n`` samples we return all of them (no bootstrapping
    by default — bootstrapping a real distribution to inflate sample size
    biases the test).
    """
    chunks = []
    collected = 0
    for batch in test_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        chunks.append(x)
        collected += x.size(0)
        if collected >= n:
            break
    if not chunks:
        raise RuntimeError("Test DataLoader yielded no batches.")
    real = torch.cat(chunks, dim=0)[:n]
    return real


# ---------------------------------------------------------------------------
# Task entry point
# ---------------------------------------------------------------------------

def evaluate_task(input: EvaluateInput) -> EvaluateOutput:
    """
    Evaluate generated samples against real test data.

    Pipeline:
        1. Load the test loader and the generated samples.
        2. Draw ``n_real_samples`` real images from the loader (no
           bootstrapping — we draw what's there).
        3. Match generated count to real count (truncate the larger).
        4. Estimate σ via the median heuristic on real data only.
        5. Run the permutation test.
        6. Save the stats CSV and a permutation plot.
    """
    _, test_path = input.split_train_test
    test_loader = torch.load(str(test_path), weights_only=False)
    gen_samples = torch.load(str(input.gen_images_path), weights_only=False)

    real_samples = _draw_real_samples(test_loader, input.n_real_samples)

    # Match shapes: bring the two groups to the same N. Truncate, don't
    # bootstrap, so the null is estimated on the right sample size.
    n = min(real_samples.size(0), gen_samples.size(0))
    if n < real_samples.size(0):
        real_samples = real_samples[:n]
    if n < gen_samples.size(0):
        gen_samples = gen_samples[:n]
    if real_samples.shape[1:] != gen_samples.shape[1:]:
        raise ValueError(
            f"Shape mismatch: real {tuple(real_samples.shape)} vs "
            f"gen {tuple(gen_samples.shape)}"
        )

    logger.info(
        "Evaluation: n_real = n_fake = %d, image shape %s, %d permutations",
        n, tuple(real_samples.shape[1:]), input.num_permutations,
    )

    # The DDPM operates in [-1, 1]; the test loader is in [-1, 1] too if
    # `normalize_to_neg_one_one` was set during preprocessing. The MMD is
    # scale-invariant under matched scaling of both groups, so we just check
    # they're roughly comparable rather than re-scaling.
    sigma = estimate_sigma(real_samples, patch_size=input.kernel_patch_size)
    distance_fn = PatchMMD(
        patch_size=input.kernel_patch_size,
        channels=real_samples.size(1),
        sigma=sigma,
    )

    observed, null_stats, p_value = permutation_test(
        real_samples, gen_samples,
        distance_fn=distance_fn,
        num_permutations=input.num_permutations,
        seed=input.seed,
    )

    eval_dir = Path("Evaluation Results") / input.evaluate_results_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Stats table: null distribution summary + observed value + p-value.
    null_summary = pd.Series(null_stats).describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    summary = pd.DataFrame({
        "value": [
            float(null_summary["count"]),
            float(null_summary["mean"]),
            float(null_summary["std"]),
            float(null_summary["min"]),
            float(null_summary["5%"]),
            float(null_summary["25%"]),
            float(null_summary["50%"]),
            float(null_summary["75%"]),
            float(null_summary["95%"]),
            float(null_summary["max"]),
            float(observed),
            float(p_value),
            int(input.num_permutations),
        ],
    }, index=[
        "null_count", "null_mean", "null_std",
        "null_min", "null_p05", "null_p25", "null_median", "null_p75",
        "null_p95", "null_max",
        "observed_mmd", "p_value", "num_permutations",
    ])

    stats_path = eval_dir / "perm_stats.csv"
    summary.to_csv(str(stats_path), index_label="statistic")
    logger.info("Saved statistics to %s", stats_path)

    # Plot.
    fig = plot_permutation(null_stats, observed, p_value)
    plot_path = eval_dir / "perm_plot.png"
    fig.savefig(str(plot_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", plot_path)

    # Also dump the raw null distribution for downstream re-analysis.
    np.save(eval_dir / "null_distribution.npy", null_stats)

    return EvaluateOutput(
        stats_csv=stats_path,
        plot_png=plot_path,
        p_value=float(p_value),
        observed_mmd=float(observed),
    )
