from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.rdx import (
    MNIST_CANDIDATES,
    RDX,
    RDXConfig,
    label_all_explanations,
)


def run_single(A, B, gamma_val, label):
    rdx = RDX(RDXConfig(gamma=gamma_val))
    result_ab, result_ba = rdx.fit_both_directions(A, B)

    print(
        f"  {label:28s}  "
        f"BSR(A->B)={result_ab.bsr:.3f}  "
        f"BSR(B->A)={result_ba.bsr:.3f}  "
        f"gamma_used={result_ab.gamma_used:.4f}"
    )
    return result_ab.bsr, result_ba.bsr


def save_comparison_plot(results: dict, save_path: Path) -> None:
    labels = list(results.keys())
    bsr_ab = [results[k][0] for k in labels]
    bsr_ba = [results[k][1] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, bsr_ab, width, label="BSR(A->B)")
    bars2 = ax.bar(x + width / 2, bsr_ba, width, label="BSR(B->A)")

    ax.set_ylabel("BSR")
    ax.set_title("Phase 3: Adaptive gamma vs fixed gamma (BSR comparison)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=9,
        )

    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison plot: {save_path}")


def main() -> None:
    base = Path("data/mnist_rdx")

    embeddings_a_path = base / "model_epoch_1_embeddings.npy"
    embeddings_b_path = base / "model_epoch_5_embeddings.npy"
    images_path = base / "model_epoch_1_images.npy"
    labels_path = base / "model_epoch_1_labels.npy"

    if not embeddings_a_path.exists():
        print("Embeddings not found. Run: python train_mnist_rdx.py")
        return

    A = np.load(embeddings_a_path).astype(np.float64)
    B = np.load(embeddings_b_path).astype(np.float64)
    images = np.load(images_path)
    digit_indices = np.load(labels_path)

    print("=" * 60)
    print("Phase 3 Experiment: Adaptive gamma vs fixed gamma")
    print("=" * 60)

    configs = [
        (0.05, "gamma=0.05 (paper)"),
        (0.10, "gamma=0.10 (paper)"),
        (10.0, "gamma=10.0 (original default)"),
        (None, "gamma=adaptive (ours)"),
    ]

    results = {}
    for gamma_val, label in configs:
        results[label] = run_single(A, B, gamma_val, label)

    print("\nSummary table:")
    print(f"{'Config':<30} {'BSR(A->B)':>10} {'BSR(B->A)':>10}")
    print("-" * 54)
    for label, (ab, ba) in results.items():
        print(f"{label:<30} {ab:>10.3f} {ba:>10.3f}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save BSR table
    with open(output_dir / "phase3_bsr_comparison.txt", "w", encoding="utf-8") as f:
        f.write(f"{'Config':<30} {'BSR(A->B)':>10} {'BSR(B->A)':>10}\n")
        f.write("-" * 54 + "\n")
        for label, (ab, ba) in results.items():
            f.write(f"{label:<30} {ab:>10.3f} {ba:>10.3f}\n")

    # Save comparison plot
    save_comparison_plot(results, output_dir / "phase3_bsr_comparison.png")

    # CLIP labeling on adaptive gamma result
    print("\n" + "=" * 60)
    print("CLIP Auto-Labeling of Explanation Grids (adaptive gamma)")
    print("=" * 60)

    try:
        rdx_adaptive = RDX(RDXConfig(gamma=None))
        result_ab, result_ba = rdx_adaptive.fit_both_directions(A, B)

        print("\nRDX(A->B) — concepts unique to epoch-1 model:")
        labels_ab = label_all_explanations(
            images, result_ab.explanations, MNIST_CANDIDATES
        )
        for i, (expl, (lbl, score)) in enumerate(
            zip(result_ab.explanations, labels_ab), start=1
        ):
            digit_labels = [["3", "5", "8"][x] for x in digit_indices[expl.indices].tolist()]
            print(
                f"  Explanation {i}: '{lbl}' (score={score:.3f}) "
                f"| actual digits={digit_labels}"
            )

        print("\nRDX(B->A) — concepts unique to epoch-5 model:")
        labels_ba = label_all_explanations(
            images, result_ba.explanations, MNIST_CANDIDATES
        )
        for i, (expl, (lbl, score)) in enumerate(
            zip(result_ba.explanations, labels_ba), start=1
        ):
            digit_labels = [["3", "5", "8"][x] for x in digit_indices[expl.indices].tolist()]
            print(
                f"  Explanation {i}: '{lbl}' (score={score:.3f}) "
                f"| actual digits={digit_labels}"
            )

        with open(output_dir / "phase3_clip_labels.txt", "w", encoding="utf-8") as f:
            f.write("RDX(A->B) CLIP Labels\n")
            f.write("-" * 40 + "\n")
            for i, (lbl, score) in enumerate(labels_ab, start=1):
                f.write(f"Explanation {i}: {lbl} (score={score:.3f})\n")

            f.write("\nRDX(B->A) CLIP Labels\n")
            f.write("-" * 40 + "\n")
            for i, (lbl, score) in enumerate(labels_ba, start=1):
                f.write(f"Explanation {i}: {lbl} (score={score:.3f})\n")

        print(f"\nCLIP labels saved to: {output_dir / 'phase3_clip_labels.txt'}")

    except ImportError as e:
        print(f"Skipping CLIP labeling: {e}")

    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()