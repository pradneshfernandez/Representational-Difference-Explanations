from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF

from src import RDX, RDXConfig
from src.utils import pairwise_rank_distances    

def plot_nmf_vs_rdx(A: np.ndarray, B: np.ndarray, images: np.ndarray, save_dir: Path):
    print("Generating NMF vs RDX Comparison Plot...")
    DA_rank = pairwise_rank_distances(A)
    DB_rank = pairwise_rank_distances(B)
    
    diff = np.clip(DB_rank - DA_rank, 0, None)
    
    num_components = 3
    nmf = NMF(n_components=num_components, random_state=42, init='nndsvda', max_iter=500)
    W = nmf.fit_transform(diff)
    
    nmf_explanations = []
    for i in range(num_components):
        top_indices = np.argsort(W[:, i])[::-1][:9]
        nmf_explanations.append(top_indices)
        
    rdx = RDX(RDXConfig(num_explanations=num_components, explanation_size=9))
    result = rdx.fit_direction(A, B, DA_rank=DA_rank, DB_rank=DB_rank)
    
    rdx_explanations = [e.indices for e in result.explanations]
    
    fig, axes = plt.subplots(num_components, 2, figsize=(8, 3 * num_components))
    fig.suptitle("Method Comparison: NMF (Generic) vs RDX (Semantic)", fontsize=16)
    
    for i in range(num_components):
        nmf_imgs = images[nmf_explanations[i]].squeeze()
        canvas_nmf = np.zeros((28*3, 28*3))
        for j, img in enumerate(nmf_imgs):
            if j >= 9: break
            r, c = j // 3, j % 3
            canvas_nmf[r*28:(r+1)*28, c*28:(c+1)*28] = img
        axes[i, 0].imshow(canvas_nmf, cmap='gray')
        axes[i, 0].set_title(f"NMF Component {i+1}")
        axes[i, 0].axis('off')
        
        if i < len(rdx_explanations):
            rdx_imgs = images[rdx_explanations[i]].squeeze()
            canvas_rdx = np.zeros((28*3, 28*3))
            for j, img in enumerate(rdx_imgs):
                if j >= 9: break
                r, c = j // 3, j % 3
                canvas_rdx[r*28:(r+1)*28, c*28:(c+1)*28] = img
            axes[i, 1].imshow(canvas_rdx, cmap='gray')
            axes[i, 1].set_title(f"RDX Explanation {i+1}")
            axes[i, 1].axis('off')
        else:
            axes[i, 1].axis('off')
            
    plt.tight_layout()
    plt.savefig(save_dir / "nmf_vs_rdx_baseline.png")
    plt.close()

def plot_bsr_ablation(A: np.ndarray, B: np.ndarray, save_dir: Path):
    print("Generating BSR Ablation Study Plot...")
    sizes = [3, 5, 9, 15, 20]
    bsr_ab = []
    bsr_ba = []
    
    DA_rank = pairwise_rank_distances(A)
    DB_rank = pairwise_rank_distances(B)
    
    for k in sizes:
        rdx = RDX(RDXConfig(explanation_size=k))
        res_ab = rdx.fit_direction(A, B, DA_rank=DA_rank, DB_rank=DB_rank)
        bsr_ab.append(res_ab.bsr)
        
        res_ba = rdx.fit_direction(B, A, DA_rank=DB_rank, DB_rank=DA_rank)
        bsr_ba.append(res_ba.bsr)

    plt.figure(figsize=(7, 5))
    plt.plot(sizes, bsr_ab, marker='o', label='RDX(A, B)', linewidth=2)
    plt.plot(sizes, bsr_ba, marker='s', label='RDX(B, A)', linewidth=2)
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance baseline')
    
    plt.title("Ablation Study: BSR vs. Explanation Size (k)", fontsize=14)
    plt.xlabel("Explanation Size (Number of Images per Grid)", fontsize=12)
    plt.ylabel("Binary Success Rate (BSR)", fontsize=12)
    plt.ylim(0.4, 1.0)
    plt.xticks(sizes)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "bsr_ablation_plot.png")
    plt.close()

if __name__ == "__main__":
    base = Path("data/mnist_rdx")
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    
    if not (base / "model_epoch_1_embeddings.npy").exists():
        print("Run `train_mnist_rdx.py` first to generate embeddings.")
        exit(1)
        
    A = np.load(base / "model_epoch_1_embeddings.npy").astype(np.float64)
    B = np.load(base / "model_epoch_5_embeddings.npy").astype(np.float64)
    images = np.load(base / "model_epoch_1_images.npy")
    
    plot_nmf_vs_rdx(A, B, images, out)
    plot_bsr_ablation(A, B, out)
    
    print(f"Visualizations successfully saved to {out.resolve()}!")
