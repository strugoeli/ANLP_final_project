import numpy as np
from cuml.manifold import UMAP,TSNE
from cuml.cluster import HDBSCAN
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UMAP_CONFIG = {'n_neighbors':15, 'n_components':5, 'metric':'cosine'}
HDBSCAN_CONFIG = {'min_cluster_size':30, 'metric':'euclidean', 'cluster_selection_method':'eom'}

class Clustering:
    def __init__(self, model,tokenizer,umap_config=None,hdbscan_config=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer

        if  umap_config is None:
            umap_config = UMAP_CONFIG

        if hdbscan_config is None:
            hdbscan_config = HDBSCAN_CONFIG

        self.umap_model = UMAP(**umap_config)
        self.cluser_model = HDBSCAN(**hdbscan_config)


    def get_features(self, dataset, max_length=512, batch_size=16):
        self.model.eval()
        features = []

        dataloader = DataLoader(dataset, batch_size=batch_size)
        for batch in tqdm(dataloader,desc='Get embeddings...'):
            inputs = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = self.model.base_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()  # CLS token
            features.append(cls_embeddings)

        features = np.concatenate(features)
        return features
    
    def run_clustering(self, embeddings, min_cluster_size=30, min_samples=None, n_neighbors=15, n_components=5):



        umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
        umap_embeddings = umap_model.fit_transform(embeddings)
        self.umap_model = umap_embeddings

        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
        clusterer = hdbscan_model.fit(umap_embeddings)
        self.cluster_model = clusterer

        return clusterer.labels_
    
    def predict_cluster(self, instances):
        inputs = self.tokenizer(instances, truncation=True, padding=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.base_model(**inputs.to(device))
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        umap_embeddings = self.umap_model.transform(embeddings)
        clusterer = self.cluster_model
        cluster = clusterer.approximate_predict(umap_embeddings)
        return cluster
