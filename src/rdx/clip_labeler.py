from __future__ import annotations

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# MNIST-specific candidate labels — extend for other datasets
MNIST_CANDIDATES = [
    "digit three",
    "digit five", 
    "digit eight",
    "ambiguous handwritten digit",
    "unclear handwriting style",
    "similar looking digits",
    "confusable digit pair",
]


def label_explanation_with_clip(
    images: Array,
    indices: List[int],
    candidate_labels: List[str],
) -> Tuple[str, float]:
    """
    Assign a human-readable label to one explanation grid using CLIP
    zero-shot similarity.

    For each explanation grid produced by RDX, we pass all images through
    CLIP's image encoder, average their embeddings into a single concept
    vector, then find the candidate text label whose CLIP text embedding
    is most similar.

    Research gap addressed:
        RDX produces image grids but no semantic labels. Users must visually
        inspect every grid. The paper authors used ChatGPT-4o manually
        (Appendix A.3). This function automates that step without any
        manual intervention.

    Reference:
        Radford et al. (2021), 'Learning Transferable Visual Models From
        Natural Language Supervision' (CLIP), ICML.
        open_clip: https://github.com/mlfoundations/open_clip

    Args:
        images:
            All dataset images as a numpy array.
            Shape (N, 1, H, W) for grayscale or (N, 3, H, W) for RGB.
            Values expected in [0, 1].
        indices:
            Indices into `images` for this explanation grid.
        candidate_labels:
            List of text strings to score against.

    Returns:
        best_label: The candidate label with highest CLIP similarity.
        best_score: The cosine similarity score (higher = more confident).
    """
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError(
            "open_clip_torch is required for CLIP labeling.\n"
            "Install it with: pip install open-clip-torch"
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    pil_images = []
    for idx in indices:
        img = images[idx]

        # handle both (1, H, W) and (H, W) shapes
        if img.ndim == 3:
            img = img.squeeze(0)          # (H, W)

        # scale to [0, 255] uint8
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

        # CLIP expects RGB — convert grayscale to RGB
        pil_img = Image.fromarray(img_uint8).convert("RGB")
        pil_images.append(preprocess(pil_img))

    image_tensor = torch.stack(pil_images)        # (grid_size, 3, 224, 224)
    text_tokens = tokenizer(candidate_labels)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # normalise both
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # mean-pool the grid into one concept vector
        mean_image_feature = image_features.mean(dim=0, keepdim=True)
        mean_image_feature = mean_image_feature / mean_image_feature.norm(
            dim=-1, keepdim=True
        )

        similarities = (mean_image_feature @ text_features.T).squeeze()

    best_idx = int(similarities.argmax().item())
    best_score = float(similarities[best_idx].item())
    best_label = candidate_labels[best_idx]

    return best_label, best_score


def label_all_explanations(
    images: Array,
    explanations,                     # List[Explanation] from explanation_sampling.py
    candidate_labels: List[str],
) -> List[Tuple[str, float]]:
    """
    Run CLIP labeling on every explanation grid in a result.

    Args:
        images:       Full image array, shape (N, 1, H, W).
        explanations: List of Explanation objects from RDXResult.
        candidate_labels: Text candidates to score against.

    Returns:
        List of (label, score) tuples, one per explanation, in the same order.
    """
    results = []
    for expl in explanations:
        label, score = label_explanation_with_clip(
            images, expl.indices, candidate_labels
        )
        results.append((label, score))
    return results