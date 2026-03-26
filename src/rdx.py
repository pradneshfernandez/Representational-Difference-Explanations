from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .clustering import spectral_cluster_affinity
from .explanation_sampling import Explanation, extract_explanations_from_clusters
from .utils import (
    binary_success_rate,
    difference_to_affinity,
    locally_biased_difference,
    pairwise_rank_distances,
    validate_embeddings,
)

Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]


@dataclass
class RDXConfig:
    gamma: float = 10.0
    beta: float = 5.0
    num_explanations: int = 5
    explanation_size: int = 9
    metric: str = "euclidean"
    random_state: int = 0
    assign_labels: str = "kmeans"


@dataclass
class RDXResult:
    normalized_distance_A: Array
    normalized_distance_B: Array
    difference_matrix: Array
    affinity_matrix: Array
    cluster_labels: IndexArray
    kept_cluster_ids: List[int]
    explanations: List[Explanation]
    bsr: float


class RDX:
    """Main class for running Representational Difference Explanations."""

    def __init__(self, config: Optional[RDXConfig] = None):
        self.config = config or RDXConfig()

    def fit_direction(self, A: Array, B: Array) -> RDXResult:
        """Run directional RDX(A, B).

        Finds groups of samples that are closer in A than in B.
        """
        A, B = validate_embeddings(A, B)
        cfg = self.config

        DA_rank = pairwise_rank_distances(A, metric=cfg.metric)
        DB_rank = pairwise_rank_distances(B, metric=cfg.metric)

        G = locally_biased_difference(DA_rank, DB_rank, gamma=cfg.gamma)
        F = difference_to_affinity(G, beta=cfg.beta, symmetrize=True)
        labels, kept_cluster_ids, _cluster_means = spectral_cluster_affinity(
            F,
            num_explanations=cfg.num_explanations,
            random_state=cfg.random_state,
            assign_labels=cfg.assign_labels,
        )

        explanations = extract_explanations_from_clusters(
            F,
            labels,
            kept_cluster_ids,
            explanation_size=cfg.explanation_size,
            max_explanations=cfg.num_explanations,
        )

        explanation_indices = [e.indices for e in explanations]
        bsr = binary_success_rate(explanation_indices, DA_rank, DB_rank)

        return RDXResult(
            normalized_distance_A=DA_rank,
            normalized_distance_B=DB_rank,
            difference_matrix=G,
            affinity_matrix=F,
            cluster_labels=labels,
            kept_cluster_ids=kept_cluster_ids,
            explanations=explanations,
            bsr=bsr,
        )

    def fit_both_directions(self, A: Array, B: Array) -> Tuple[RDXResult, RDXResult]:
        """Run both RDX(A, B) and RDX(B, A)."""
        result_ab = self.fit_direction(A, B)
        result_ba = self.fit_direction(B, A)
        return result_ab, result_ba