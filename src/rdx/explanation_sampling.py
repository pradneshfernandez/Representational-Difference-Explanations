from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .utils import mean_within_cluster_affinity, top_k_neighbors_within_cluster

Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]


@dataclass
class Explanation:
    """
    One explanation grid produced from one retained cluster.

    Attributes:
        anchor_index: The central sample selected by K-neighborhood affinity.
        indices: Final explanation members, usually anchor + top-k neighbors.
        cluster_indices: All indices belonging to the underlying cluster.
        score: K-neighborhood affinity score for the chosen anchor.
        cluster_mean_affinity: Mean within-cluster affinity for the parent cluster.
    """
    anchor_index: int
    indices: List[int]
    cluster_indices: List[int]
    score: float
    cluster_mean_affinity: float


def extract_explanations_from_clusters(
    F: Array,
    labels: IndexArray,
    kept_cluster_ids: Sequence[int],
    explanation_size: int,
    max_explanations: Optional[int] = None,
) -> List[Explanation]:
    """
    Extract one explanation grid from each retained cluster using
    K-neighborhood affinity (KNA).

    Method:
    - For each retained cluster:
        - try every point as an anchor
        - find its top-k neighbors within that cluster using affinity values
        - compute KNA score = sum of anchor-to-neighbor affinities
        - keep the anchor with the maximum score
    - Return explanations sorted by cluster mean affinity, descending

    Args:
        F:
            Symmetric affinity matrix of shape (n, n).
        labels:
            Cluster label for each sample, shape (n,).
        kept_cluster_ids:
            Cluster ids retained after dropping the weakest cluster.
        explanation_size:
            Number of samples to include in each explanation grid.
            Must be at least 2.
        max_explanations:
            Optional cap on number of explanations returned.

    Returns:
        List[Explanation]
    """
    if explanation_size < 2:
        raise ValueError("explanation_size must be at least 2.")

    k = explanation_size - 1
    candidates: List[Explanation] = []

    for cid in kept_cluster_ids:
        members = np.flatnonzero(labels == cid).astype(np.int64)
        if len(members) == 0:
            continue

        best_anchor = None
        best_neighbors: List[int] = []
        best_score = -float("inf")

        for anchor in members.tolist():
            neighbors = top_k_neighbors_within_cluster(
                affinity_row=F[anchor],
                cluster_members=members.tolist(),
                anchor=anchor,
                k=k,
            )

            score = float(np.sum(F[anchor, neighbors])) if neighbors else -float("inf")

            if score > best_score:
                best_score = score
                best_anchor = anchor
                best_neighbors = neighbors

        if best_anchor is None:
            continue

        cluster_mean = mean_within_cluster_affinity(F, members.tolist())

        candidates.append(
            Explanation(
                anchor_index=int(best_anchor),
                indices=[int(best_anchor)] + [int(x) for x in best_neighbors],
                cluster_indices=[int(x) for x in members.tolist()],
                score=float(best_score),
                cluster_mean_affinity=float(cluster_mean),
            )
        )

    candidates.sort(key=lambda x: x.cluster_mean_affinity, reverse=True)

    if max_explanations is not None:
        candidates = candidates[:max_explanations]

    return candidates