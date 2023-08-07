from cuml.preprocessing import normalize
import  cupy as cp
import torch
from cuml.manifold import UMAP
from cuml.cluster import KMeans
import joblib
import os
import json
from typing import List
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UMAP_CONFIG = {'n_neighbors': 27,
                'n_components': 50,
                'metric': 'euclidean',
                'n_epochs': 1000,
                'min_dist':0.0,
                'spread':1.8,
                'verbose':True,
                'random_state':42}

KMeans_CONFIG = {'n_clusters': 80,
                 'max_iter': 300,
                 'random_state': 42,
                 'verbose':2}


class EmbeddingClusterer:

    """
    This is a class that applies dimensionality reduction and clustering on embeddings using UMAP and KMeans models.
    The embeddings are first normalized, then processed with UMAP for dimensionality reduction,
    and subsequently clustered using KMeans.

    Attributes:
    umap_model (UMAP): The UMAP model used for dimensionality reduction.
    cluster_predictor (KMeans): The KMeans model used for clustering.
    umap_config (dict): The configuration for the UMAP model.
    cluster_predictor_config (dict): The configuration for the KMeans model.
    """
    def __init__(self, umap_model=None, cluster_predictor=None, umap_config=None, cluster_predictor_config=None):

        if umap_config is None:
            umap_config = UMAP_CONFIG
        if cluster_predictor_config is None:
            cluster_predictor_config = KMeans_CONFIG

        self.umap_config = umap_config
        self.cluster_predictor_config = cluster_predictor_config
        self.umap_model = umap_model
        self.cluster_predictor = cluster_predictor

    def run_clustering(self, embeddings: np.ndarray) -> List[int]:
        """
        This method runs the clustering on the embeddings and returns the cluster labels for each embedding.

        Parameters:
        embeddings (np.ndarray): An array of embeddings. Each embedding is a vector in high-dimensional space.

        Returns:
        List[int]: A list of cluster labels, each label corresponds to the input embeddings in the same order.
        """
        # Using L2 normalization
        normalized_embeddings = normalize(cp.array(embeddings))
        self.umap_model = UMAP(**UMAP_CONFIG).fit(normalized_embeddings)
        umap_embd = self.umap_model.transform(normalized_embeddings)
        self.cluster_predictor = KMeans(**KMeans_CONFIG).fit(umap_embd)
        clusters = self.cluster_predictor.labels_
        return clusters.tolist()

    def predict_cluster(self, embeddings: np.ndarray) -> List[int]:
        """
        This method predicts the cluster labels for given embeddings based on the previously trained UMAP and KMeans model.

        Parameters:
        embeddings (np.ndarray): An array of embeddings. Each embedding is a vector in high-dimensional space.

        Returns:
        List[int]: A list of predicted cluster labels, each label corresponds to the input embeddings in the same order.
        """
        # Using L2 normalization
        normalized_embeddings = normalize(cp.array(embeddings))
        umap_embd = self.umap_model.transform(normalized_embeddings)
        cluster_labels = self.cluster_predictor.predict(umap_embd)
        return cluster_labels.tolist()

    def save_umap_and_cluster_predictor(self, path: str):
        """
        This method saves the UMAP and KMeans models to the given path.
        """
        os.makedirs(path, exist_ok=True)
        self.save_umap_model(path)
        self.save_cluster_predictor_model(path)

    def save_umap_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.umap_model, path + '/umap_model.joblib')
        with open(path + '/umap_config.json', 'w') as f:
            json.dump(self.umap_config, f)

    def save_cluster_predictor_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.cluster_predictor, path + '/cluster_predictor_model.joblib')
        with open(path + '/cluster_predictor_config.json', 'w') as f:
            json.dump(self.cluster_predictor_config, f)

    def load_umap_model(self, path: str):

        model_path = os.path.join(path, 'umap_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path {model_path} does not exist")

        self.umap_model = joblib.load(model_path)
        return  self.umap_model

    def load_cluster_predictor_model(self, path: str):

        model_path = os.path.join(path, 'cluster_predictor_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path {model_path} does not exist")

        self.cluster_predictor = joblib.load(model_path)

        return self.cluster_predictor

    def get_centroids(self):
        return self.cluster_predictor.cluster_centers_

