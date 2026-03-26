from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from src import RDX, RDXConfig


def show_explanation_grid(
    images: np.ndarray,
    indices: list[int],
    title: str,
    save_path: Path,
    cols: int = 3,
) -> None:
    rows = int(np.ceil(len(indices) / cols))
    plt.figure(figsize=(cols * 2, rows * 2))

    for plot_idx, data_idx in enumerate(indices, start=1):
        plt.subplot(rows, cols, plot_idx)
        img = images[data_idx].squeeze()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(str(data_idx))

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_explanations_to_json(
    result,
    labels: np.ndarray,
    save_path: Path,
) -> None:
    data = []
    for i, expl in enumerate(result.explanations, start=1):
        data.append(
            {
                "explanation_id": i,
                "indices": expl.indices,
                "labels": labels[expl.indices].tolist(),
                "anchor_index": expl.anchor_index,
                "score": float(expl.score),
                "cluster_mean_affinity": float(expl.cluster_mean_affinity),
            }
        )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_bsr_plot(result_ab, result_ba, save_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    labels_plot = ["A→B", "B→A"]
    values = [result_ab.bsr, result_ba.bsr]

    plt.bar(labels_plot, values)
    plt.title("BSR Comparison")
    plt.ylabel("BSR")
    plt.ylim(0, 1)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    base = Path("data/mnist_rdx")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    A = np.load(base / "model_epoch_1_embeddings.npy").astype(np.float64)
    B = np.load(base / "model_epoch_5_embeddings.npy").astype(np.float64)
    labels = np.load(base / "model_epoch_1_labels.npy")
    images = np.load(base / "model_epoch_1_images.npy")

    rdx = RDX(
        RDXConfig(
            gamma=10.0,
            beta=5.0,
            num_explanations=5,
            explanation_size=9,
            metric="euclidean",
            random_state=0,
        )
    )

    result_ab, result_ba = rdx.fit_both_directions(A, B)

    print("=" * 80)
    print("RDX(A, B): epoch 1 vs epoch 5")
    print("BSR(A, B):", result_ab.bsr)
    for i, expl in enumerate(result_ab.explanations, start=1):
        print(f"  Explanation {i}: {expl.indices}")
        print("    labels:", labels[expl.indices].tolist())

    print("\n" + "=" * 80)
    print("RDX(B, A): epoch 5 vs epoch 1")
    print("BSR(B, A):", result_ba.bsr)
    for i, expl in enumerate(result_ba.explanations, start=1):
        print(f"  Explanation {i}: {expl.indices}")
        print("    labels:", labels[expl.indices].tolist())

    # Save BSR values
    with open(output_dir / "bsr.txt", "w", encoding="utf-8") as f:
        f.write(f"BSR(A,B): {result_ab.bsr}\n")
        f.write(f"BSR(B,A): {result_ba.bsr}\n")

    # Save explanations as JSON
    save_explanations_to_json(result_ab, labels, output_dir / "rdx_ab.json")
    save_explanations_to_json(result_ba, labels, output_dir / "rdx_ba.json")

    # Save BSR plot
    save_bsr_plot(result_ab, result_ba, output_dir / "bsr_plot.png")

    # Save first explanation grid in both directions
    if result_ab.explanations:
        show_explanation_grid(
            images,
            result_ab.explanations[0].indices,
            "RDX(A, B) - Explanation 1",
            output_dir / "RDX_A_B_Explanation_1.png",
        )

    if result_ba.explanations:
        show_explanation_grid(
            images,
            result_ba.explanations[0].indices,
            "RDX(B, A) - Explanation 1",
            output_dir / "RDX_B_A_Explanation_1.png",
        )

    print("\nSaved outputs to:", output_dir.resolve())


if __name__ == "__main__":
    main()