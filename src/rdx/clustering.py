from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import SpectralClustering

from .utils import mean_within_cluster_affinity

Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]


def spectral_cluster_affinity(
    F: Array,
    num_explanations: int,
    random_state: int = 0,
    assign_labels: str = "kmeans",
) -> Tuple[IndexArray, List[int], Dict[int, float]]:
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]

    if F.shape != (n, n):
        raise ValueError("F must be a square affinity matrix.")

    n_clusters = max(2, num_explanations + 1)
    if n_clusters >= n:
        raise ValueError("Too many clusters for the number of samples.")

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels=assign_labels,
        random_state=random_state,
    )
    labels = clustering.fit_predict(F).astype(np.int64)

    cluster_ids = sorted(np.unique(labels).tolist())
    cluster_means: Dict[int, float] = {}

    for cid in cluster_ids:
        members = np.flatnonzero(labels == cid).tolist()
        cluster_means[cid] = mean_within_cluster_affinity(F, members)

    drop_cid = min(cluster_means, key=cluster_means.get)
    kept_cluster_ids = [cid for cid in cluster_ids if cid != drop_cid]

    return labels, kept_cluster_ids, cluster_means