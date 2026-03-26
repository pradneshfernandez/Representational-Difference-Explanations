# Representational-Difference-Explanations

# Representational Difference Explanations (RDX) - Course Project

## 📖 About This Project
This repository contains the implementation, demonstration, and proposed improvements for **Representational Difference Explanations (RDX)**, a post-hoc Explainable AI (XAI) method introduced at NeurIPS 2025. 

While comparison is a cornerstone of scientific analysis, existing Dictionary Learning (DL) XAI tools—like Sparse Autoencoders (SAEs) or Non-Negative Matrix Factorization (NMF)—struggle to effectively compare two machine learning models. When models are highly similar, DL methods extract identical concepts for both, masking subtle performance differences. RDX solves this by directly comparing two representations against each other to isolate their differences, identifying inputs that only one of the two models considers to be semantically related.

---

## 🚀 Project Structure

### Phase 1: Core Implementation
The core algorithm takes an input dataset of $n$ items and extracts "difference explanation" grids through four steps:
1. **Compute Distances:** Calculate pairwise Euclidean distances for embeddings from Model A and Model B.
2. **Normalize:** Convert raw distances into scale-invariant nearest-neighbor ranks.
3. **Difference Matrix:** Compute a locally-biased difference matrix that prioritizes items where at least one model considers the embeddings to be highly similar.
4. **Spectral Clustering & KNA:** Apply spectral clustering to an affinity matrix, then use the K-neighborhood affinity (KNA) to sample discrete grids of 9-25 images that visualize the conceptual differences.
*(Note: A Centered Kernel Alignment (CKA) step is also available to structurally align completely unaligned models before computing distances).*

### Phase 2: Demonstration & Case Study
To demonstrate why RDX is necessary, we use a modified **MNIST-** dataset.
* We train a 2-layer CNN and save an early "strong" checkpoint (95% accuracy) and a final "expert" checkpoint (98% accuracy).
* Running baselines like NMF, SAE, and KMeans produces indistinguishable explanations that fail to explain the 3% performance gap. 
* Running RDX successfully isolates the exact styles of 3s, 5s, and 8s that the weaker model confuses but the expert model effectively separates.

*Further dataset experiments (CUB, iNaturalist, ImageNet) are detailed in the `experiments/` folder.*

### Phase 3: Proposed Improvements (Future Work)
This project identifies and proposes solutions for several structural limitations of the original RDX method:
* **Memory Bottleneck:** The current approach requires $\mathcal{O}(n^2)$ memory to compute and store the full pairwise distance matrix, limiting scalability. We propose localized approximation techniques to scale beyond the 5,000 data points tested in the original study.
* **Beyond Discrete Concepts:** RDX currently defines a concept strictly as a discrete grid of highly related images. We aim to extend this to extract **continuous concepts** (e.g., a continuous spectrum of "roundness") that vary linearly.
* **Alternative Distance Metrics:** RDX is currently reliant on Euclidean distance, which may misrepresent distances along complex, non-linear data manifolds. We plan to experiment with geodesic distance metrics.
* **Cross-Modal Capabilities:** Adapting the visual-centric RDX framework to compare Large Language Models (LLMs) via text and multi-modal representations.

---
# Representational Difference Explanations (RDX)

## Overview
This project implements the RDX method to compare representations learned by neural networks.

## Objective
To reproduce key experiments from the RDX paper using MNIST dataset.

## Method
- Train a CNN on MNIST digits (3,5,8)
- Extract embeddings at:
  - Epoch 1 (early representation)
  - Epoch 5 (trained representation)
- Apply RDX to identify representational differences

## Results

### BSR Scores
- RDX(A, B): 0.88
- RDX(B, A): 0.80

### Observations
- Early model groups digits based on rough visual similarity
- Later model learns more structured and class-specific features
- RDX successfully identifies ambiguous samples between classes (e.g., 3 vs 5)

## Files
- `train_mnist_rdx.py` → training and embedding extraction
- `run_mnist_rdx.py` → RDX analysis
- `src/` → core RDX implementation
- `outputs/` → saved explanations and plots

## Conclusion
RDX effectively highlights how model representations evolve during training and identifies where models differ in understanding data.
---

## 📚 References
*  Kondapaneni, N., Mac Aodha, O., & Perona, P. (2025). *Representational Difference Explanations*. 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
