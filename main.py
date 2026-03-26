from __future__ import annotations

from pathlib import Path

import numpy as np

from src import RDX, RDXConfig


def main() -> None:
    base = Path("data/mnist_rdx")

    if not (base / "model_epoch_1_embeddings.npy").exists():
        print("Embeddings not found. Run: python train_mnist_rdx.py")
        return

    A = np.load(base / "model_epoch_1_embeddings.npy").astype(np.float64)
    B = np.load(base / "model_epoch_5_embeddings.npy").astype(np.float64)

    rdx = RDX(RDXConfig())
    result_ab, result_ba = rdx.fit_both_directions(A, B)

    print("RDX(A, B) BSR:", result_ab.bsr)
    print("RDX(B, A) BSR:", result_ba.bsr)


if __name__ == "__main__":
    main()