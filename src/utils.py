from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances

Array = NDArray[np.float64]


def validate_embeddings(A: Array, B: Array) -> Tuple[Array, Array]:
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
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]

    D = pairwise_distances(X, metric=metric)
    rank_dist = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        order = np.argsort(D[i], kind="stable")
        rank = 1
        for j in order:
            if j == i:
                continue
            rank_dist[i, j] = rank
            rank += 1

    return rank_dist


def locally_biased_difference(
    DA_rank: Array,
    DB_rank: Array,
    gamma: float = 10.0,
    eps: float = 1e-12,
) -> Array:
    DA_rank = np.asarray(DA_rank, dtype=np.float64)
    DB_rank = np.asarray(DB_rank, dtype=np.float64)

    if DA_rank.shape != DB_rank.shape:
        raise ValueError("DA_rank and DB_rank must have the same shape.")

    denom = np.minimum(DA_rank, DB_rank)
    denom = np.where(denom <= 0, eps, denom)

    G = np.tanh(gamma * (DA_rank - DB_rank) / denom)
    np.fill_diagonal(G, 0.0)
    return G


def difference_to_affinity(G: Array, beta: float = 5.0, symmetrize: bool = True) -> Array:
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
    members = np.array([m for m in cluster_members if m != anchor], dtype=np.int64)
    if members.size == 0:
        return []

    vals = affinity_row[members]
    order = np.argsort(-vals, kind="stable")
    chosen = members[order[: min(k, len(order))]]
    return chosen.tolist()