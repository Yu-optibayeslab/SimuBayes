# dimensionality_reducer.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import umap.umap_ as umap

class DimensionalityReducer:
    def __init__(self):
        pass

    def pca(self, data, n_components=2):
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        return reduced_data

    def tsne(self, data, n_components=2, perplexity=30, random_state=42):
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        reduced_data = tsne.fit_transform(data)
        return reduced_data

    #def umap(self, data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    #    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    #    reduced_data = reducer.fit_transform(data)
    #    return reduced_data
