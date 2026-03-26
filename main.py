
import torch
from src.utils import get_data
from src.extract import extract_features
from src.rdx import compute_rdx
from src.clustering import cluster_features

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
loader_a = get_data(flip=False)
loader_b = get_data(flip=True)

# Models
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def feature_layer(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return x

    def forward(self, x):
        return self.fc2(self.feature_layer(x))

model_a = SimpleNN().to(device)
model_b = SimpleNN().to(device)

# NOTE: skipping training for now (can add later)

# Extract features
features_a = extract_features(model_a, loader_a, device)
features_b = extract_features(model_b, loader_b, device)

# RDX
rdx_features = compute_rdx(features_a, features_b)

# Clustering
labels = cluster_features(rdx_features)

# Plot
plt.scatter(rdx_features[:,0], rdx_features[:,1], c=labels)
plt.title("RDX Feature Differences")
plt.show()


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from src import RDX, RDXConfig


def load_embeddings(npy_path: Path) -> np.ndarray:
    """Load embeddings from a .npy file.

    Expected shape: (n_samples, dim)
    """
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {npy_path}, got shape {arr.shape}")
    return arr.astype(np.float64)


def save_indices(path: Path, indices: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, group in enumerate(indices, start=1):
            f.write(f"Explanation {idx}: {group}")
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Representational Difference Explanations (RDX)")
    parser.add_argument("--a", type=Path, required=True, help="Path to embeddings for model A (.npy)")
    parser.add_argument("--b", type=Path, required=True, help="Path to embeddings for model B (.npy)")
    parser.add_argument("--gamma", type=float, default=10.0, help="Gamma for locally-biased difference")
    parser.add_argument("--beta", type=float, default=5.0, help="Beta for difference-to-affinity conversion")
    parser.add_argument("--num-explanations", type=int, default=5, help="Number of explanations to keep")
    parser.add_argument("--explanation-size", type=int, default=9, help="Size of each explanation grid")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric")
    parser.add_argument("--random-state", type=int, default=0, help="Random state for spectral clustering")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save explanation indices and matrices",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    A = load_embeddings(args.a)
    B = load_embeddings(args.b)

    config = RDXConfig(
        gamma=args.gamma,
        beta=args.beta,
        num_explanations=args.num_explanations,
        explanation_size=args.explanation_size,
        metric=args.metric,
        random_state=args.random_state,
    )

    rdx = RDX(config)
    result_ab, result_ba = rdx.fit_both_directions(A, B)

    print("=" * 80)
    print("RDX(A, B): groups that are closer in A than in B")
    print(f"BSR(A, B): {result_ab.bsr:.4f}")
    for i, expl in enumerate(result_ab.explanations, start=1):
        print(f"  Explanation {i}: {expl.indices}")

    print("" + "=" * 80)
    print("RDX(B, A): groups that are closer in B than in A")
    print(f"BSR(B, A): {result_ba.bsr:.4f}")
    for i, expl in enumerate(result_ba.explanations, start=1):
        print(f"  Explanation {i}: {expl.indices}")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

        save_indices(args.save_dir / "explanations_ab.txt", [e.indices for e in result_ab.explanations])
        save_indices(args.save_dir / "explanations_ba.txt", [e.indices for e in result_ba.explanations])

        np.save(args.save_dir / "difference_matrix_ab.npy", result_ab.difference_matrix)
        np.save(args.save_dir / "difference_matrix_ba.npy", result_ba.difference_matrix)
        np.save(args.save_dir / "affinity_matrix_ab.npy", result_ab.affinity_matrix)
        np.save(args.save_dir / "affinity_matrix_ba.npy", result_ba.affinity_matrix)

        print(f"Saved outputs to: {args.save_dir}")


if __name__ == "__main__":
    main()
