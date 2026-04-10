import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import numpy as np
import os
import argparse
from tqdm import tqdm

def get_embeddings(model, dataloader, device):
    """
    Passes a dataset through a model to extract latent representations.

    Args:
        model (torch.nn.Module): The neural network with the classification head removed.
        dataloader (torch.utils.data.DataLoader): The data provider.
        device (torch.device): GPU or CPU.

    Returns:
        np.ndarray: A matrix of shape (n_samples, latent_dim).
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device)
            feats = model(imgs)
            embeddings.append(feats.cpu().numpy())
            
    return np.vstack(embeddings)

def get_model(model_name, pretrained=False):
    """
    Dynamically loads a vision model and strips the final layer.
    """
    weights = "DEFAULT" if pretrained else None
    # Dynamically fetch model from torchvision.models
    model_fn = getattr(models, model_name)
    model = model_fn(weights=weights)
    
    # Remove final layer to get the representation
    if hasattr(model, 'fc'):
        model.fc = nn.Identity()
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    
    return model

def run_extraction(args):
    """
    Orchestrates the feature extraction process based on command line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dynamic Dataset Loading
    if args.dataset.lower() == "cifar10":
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.dataset.lower() == "mnist":
        # MNIST needs to be converted to 3-channel for standard ImageNet models
        transform = transforms.Compose([transforms.Grayscale(3), transform])
        dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    indices = torch.arange(min(len(dataset), args.num_samples)) 
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    # Load Models
    model_a = get_model(args.model, pretrained=False).to(device)
    model_b = get_model(args.model, pretrained=True).to(device)

    print(f"Extracting {args.model} features from {args.dataset}...")
    emb_a = get_embeddings(model_a, loader, device)
    emb_b = get_embeddings(model_b, loader, device)

    os.makedirs('embeddings', exist_ok=True)
    np.save(f'embeddings/emb_a_{args.dataset}_{args.model}.npy', emb_a)
    np.save(f'embeddings/emb_b_{args.dataset}_{args.model}.npy', emb_b)
    print("Step 1 Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDX Feature Extraction")
    parser.add_argument("--model", type=str, default="resnet18", help="torchvision model name")
    parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or mnist")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of images to process")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    run_extraction(args)