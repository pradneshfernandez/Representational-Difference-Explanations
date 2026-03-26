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