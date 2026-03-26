import numpy as np
from sklearn.decomposition import PCA

class RDX:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def compute_difference(self, f1, f2):
        """
        Compute representation difference
        """
        return f1 - f2

    def fit(self, f1, f2):
        diff = self.compute_difference(f1, f2)
        self.reduced = self.pca.fit_transform(diff)
        return self.reduced

    def transform(self, f1, f2):
        diff = self.compute_difference(f1, f2)
        return self.pca.transform(diff)

    def get_top_directions(self):
        """
        Returns principal directions (important for explanation)
        """
        return self.pca.components_