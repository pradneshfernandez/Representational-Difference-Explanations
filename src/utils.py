import numpy as np

def normalize_features(f):
    return (f - np.mean(f, axis=0)) / (np.std(f, axis=0) + 1e-8)

import matplotlib.pyplot as plt

def plot_2d(features, labels=None, title=""):
    plt.figure()
    plt.scatter(features[:,0], features[:,1], c=labels)
    plt.title(title)
    plt.show()

