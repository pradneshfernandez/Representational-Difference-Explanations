from __future__ import annotations
from typing import List, Optional, Sequence, Tuple   # <-- add Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances

Array = NDArray[np.float64]


def validate_embeddings(A: Array, B: Array) -> Tuple[Array, Array]:
    # ── UNCHANGED ──────────────────────────────────────────────────────────
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays of shape (n_samples, dim).")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of samples.")
    if A.shape[0] < 3:
        raise ValueError("Need at least 3 samples for RDX.")
    return A, B


def pairwise_rank_distances(X: Array, metric: str = "euclidean") -> Array:
    # ── UNCHANGED ──────────────────────────────────────────────────────────
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    D = pairwise_distances(X, metric=metric)
    order = np.argsort(D, axis=1, kind="stable")
    rank_dist = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(rank_dist, order, np.arange(n)[None, :], axis=1)
    np.fill_diagonal(rank_dist, 0.0)
    return rank_dist


# ── NEW FUNCTION ────────────────────────────────────────────────────────────
def compute_adaptive_gamma(
    DA_rank: Array,
    DB_rank: Array,
    percentile: float = 50,
    clip_min: float = 0.01,
    clip_max: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """
    Compute gamma automatically using the median heuristic on distance ratios.

    Motivation (research gap):
        The original paper sweeps gamma in {0.05, 0.1} manually on one
        comparison per experiment group (Appendix C.4).  This is fragile:
        different embedding spaces have different distance scales, so a
        fixed gamma under- or over-saturates the tanh for other datasets.

    Method (from literature):
        The median heuristic for kernel bandwidth selection
        (Gretton et al., 2012, JMLR "A Kernel Two-Sample Test") sets the
        bandwidth to the median pairwise distance.  We adapt this idea:
        gamma = 1 / median( |D_A - D_B| / min(D_A, D_B) )
        — i.e. the inverse of the median locally-normalised gap between
        the two rank matrices.  This makes gamma scale-invariant and
        dataset-agnostic, requiring no manual sweep.

    Returns:
        A scalar gamma in [clip_min, clip_max].
    """
    DA_rank = np.asarray(DA_rank, dtype=np.float64)
    DB_rank = np.asarray(DB_rank, dtype=np.float64)

    # Use only the upper triangle to avoid counting each pair twice
    # and to exclude the zero diagonal
    n = DA_rank.shape[0]
    idx = np.triu_indices(n, k=1)
    d_A = DA_rank[idx]
    d_B = DB_rank[idx]

    # Same denominator used inside locally_biased_difference
    denom = np.minimum(d_A, d_B)
    denom = np.where(denom <= 0, eps, denom)

    ratios = np.abs(d_A - d_B) / denom

    # Only consider pairs where there is some disagreement
    nonzero = ratios[ratios > 0]
    if nonzero.size == 0:
        return float(clip_min)   # representations are identical; gamma irrelevant

    median_ratio = float(np.percentile(nonzero, percentile))
    gamma = 1.0 / (median_ratio + eps)

    return float(np.clip(gamma, clip_min, clip_max))
# ── END NEW FUNCTION ─────────────────────────────────────────────────────────


def locally_biased_difference(
    DA_rank: Array,
    DB_rank: Array,
    gamma: Optional[float] = None,    # <── CHANGED: was gamma: float = 10.0
    eps: float = 1e-12,
) -> Array:
    """
    Compute the locally-biased difference matrix G_A,B (Eq. 1 in the paper).

    Change from original:
        gamma now defaults to None.  When None, it is computed automatically
        via compute_adaptive_gamma().  Pass an explicit float to reproduce
        the original paper's behaviour exactly (e.g. gamma=0.05 or 0.1).
    """
    DA_rank = np.asarray(DA_rank, dtype=np.float64)
    DB_rank = np.asarray(DB_rank, dtype=np.float64)
    if DA_rank.shape != DB_rank.shape:
        raise ValueError("DA_rank and DB_rank must have the same shape.")

    # ── NEW: auto-select gamma if not provided ────────────────────────────
    if gamma is None:
        gamma = compute_adaptive_gamma(DA_rank, DB_rank)
        print(f"[RDX] adaptive gamma = {gamma:.4f}")
    # ── END NEW ──────────────────────────────────────────────────────────

    denom = np.minimum(DA_rank, DB_rank)
    denom = np.where(denom <= 0, eps, denom)
    G = np.tanh(gamma * (DA_rank - DB_rank) / denom)
    np.fill_diagonal(G, 0.0)
    return G


def difference_to_affinity(G: Array, beta: float = 5.0, symmetrize: bool = True) -> Array:
    # ── UNCHANGED ──────────────────────────────────────────────────────────
    G = np.asarray(G, dtype=np.float64)
    F = np.exp(-beta * G)
    if symmetrize:
        F = 0.5 * (F + F.T)
    np.fill_diagonal(F, 1.0)
    return F


def binary_success_rate(
    explanation_indices: Sequence[Sequence[int]],
    DA_rank: Array,
    DB_rank: Array,
) -> float:
    # ── UNCHANGED ──────────────────────────────────────────────────────────
    total = 0
    success = 0
    for group in explanation_indices:
        idxs = list(group)
        for i in idxs:
            for j in idxs:
                if i == j:
                    continue
                total += 1
                if DA_rank[i, j] < DB_rank[i, j]:
                    success += 1
    if total == 0:
        return float("nan")
    return success / total


def mean_within_cluster_affinity(F: Array, members: Sequence[int]) -> float:
    # ── UNCHANGED ──────────────────────────────────────────────────────────
    members = list(members)
    if len(members) <= 1:
        return float("-inf")
    sub = F[np.ix_(members, members)]
    mask = ~np.eye(len(members), dtype=bool)
    values = sub[mask]
    if values.size == 0:
        return float("-inf")
    return float(values.mean())


def top_k_neighbors_within_cluster(
    affinity_row: Array,
    cluster_members: Sequence[int],
    anchor: int,
    k: int,
) -> List[int]:
    # ── UNCHANGED ──────────────────────────────────────────────────────────
    members = np.array([m for m in cluster_members if m != anchor], dtype=np.int64)
    if members.size == 0:
        return []
    vals = affinity_row[members]
    order = np.argsort(-vals, kind="stable")
    chosen = members[order[: min(k, len(order))]]
    return chosen.tolist()