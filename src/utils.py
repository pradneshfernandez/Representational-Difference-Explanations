import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data(batch_size=64, flip=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    if flip:
        dataset.data = torch.flip(dataset.data, dims=[2])  # horizontal flip

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader