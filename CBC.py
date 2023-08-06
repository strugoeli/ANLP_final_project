import faiss
import numpy as np
import torch

import faiss
import numpy as np
import torch
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN, approximate_predict
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import joblib
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UMAP_CONFIG = {'n_neighbors': 27, 'n_components': 2, 'metric': 'cosine'}
HDBSCAN_CONFIG = {'min_cluster_size': 80, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}


def normalize_emb_z_score(embeddings):
    emb_mean = np.mean(embeddings)
    emb_std = np.std(embeddings)
    return (embeddings - emb_mean) / emb_std


class ClusterBasedClassifier:
    def __init__(self,ds, model, tokenizer, umap_config=None, hdbscan_config=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.ds = ds


        if umap_config is None:
            umap_config = UMAP_CONFIG
        if hdbscan_config is None:
            hdbscan_config = HDBSCAN_CONFIG

        self.umap_config = umap_config
        self.hdbscan_config = hdbscan_config
        self.umap_model = None
        self.cluster_predictor = None


    def predict(self, test_ds):
        pass




    def run_clustering(self, split='train'):
        embeddings =self.ds['train']['embeddings']
        normalized_embeddings = normalize_emb_z_score(embeddings)

        if self.umap_model is None:
            self.umap_model = UMAP(**self.umap_config).fit(normalized_embeddings)

        umap_embeddings = self.umap_model.transform(normalized_embeddings)

        if self.cluster_predictor is None:
            self.cluster_predictor = HDBSCAN(**self.hdbscan_config, prediction_data=True).fit(umap_embeddings)

        cluster_labels = self.cluster_predictor.labels_
        return cluster_labels

    def predict_cluster(self, embeddings):
        normalized_embeddings = normalize_emb_z_score(embeddings)

        if self.umap_model is not None:
            umap_embeddings = self.umap_model.transform(normalized_embeddings)
        else:
            raise ValueError("UMAP model is not yet initialized. Run clustering first.")

        cluster_labels = approximate_predict(self.cluster_predictor, umap_embeddings)
        return cluster_labels

    # Added saving and loading methods for UMAP and HDBSCAN:

    def save_umap_and_hdbscan_models(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.save_umap_model(path + '/umap_model.joblib')
        self.save_hdbscan_model(path + '/hdbscan_model.joblib')

    def save_umap_model(self, path: str):
        joblib.dump(self.umap_model, path)

    def save_hdbscan_model(self, path: str):
        joblib.dump(self.cluster_predictor, path)

    def load_umap_model(self, path: str):
        self.umap_model = joblib.load(path)

    def load_hdbscan_model(self, path: str):
        self.cluster_predictor = joblib.load(path)

