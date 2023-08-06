import faiss
import numpy as np
import torch

from cuml.preprocessing import normalize

import  cupy as cp
import faiss
import numpy as np
import torch
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN,approximate_predict
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import joblib
import os
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UMAP_CONFIG = {'n_neighbors': 27, 'n_components': 2, 'metric': 'cosine'}
HDBSCAN_CONFIG = {'min_cluster_size': 80, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}

def normalize_emb_z_score(embeddings):
    emb_mean = np.mean(embeddings)
    emb_std = np.std(embeddings)
    return (embeddings - emb_mean) / emb_std

class ClusterBasedClassifier:
    def __init__(self, model, tokenizer,umap_model=None,cluster_predictor=None, umap_config=None, hdbscan_config=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer

        if umap_config is None:
            umap_config = UMAP_CONFIG
        if hdbscan_config is None:
            hdbscan_config = HDBSCAN_CONFIG

        self.umap_config = umap_config
        self.hdbscan_config = hdbscan_config
        self.umap_model = umap_model
        self.cluster_predictor = cluster_predictor


    def find_closest_clusters(self, centroids_index, embedding, k):
        print(f"Embedding Shape: {embedding.shape}")  
        print(f"FAISS d Dimension: {centroids_index.d}") 

        assert embedding.shape[-1] == centroids_index.d, "Dimension mismatch between embedding and FAISS index!"

        _, I = centroids_index.search(embedding, k)
        return I[:, 0]


    @staticmethod
    def compute_class_scores(faiss_indexes, embedding):
        scores = []
        indices = []

        for cluster,index in faiss_indexes.items():
            D, I = index.search(embedding.reshape(1, -1), 1)
            scores_cluster = -D[0]
            scores.append(scores_cluster)
            indices.append(I[0])  # Store the index corresponding to each distance.

        scores_combined = np.concatenate(scores)
        indices_combined = np.concatenate(indices)

        return scores_combined, indices_combined


    def evaluate(self, test_loader, index_path, k=1, batch_size=32):
        self.model.eval()
        all_embeddings = []
        all_labels = []

        # First pass: get all embeddings and labels
        for batch in tqdm(test_loader, desc="Computing embeddings during evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.no_grad():
                outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            all_embeddings.append(embeddings)
            all_labels.append(batch['subcategory'])

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Assign clusters and handle outliers (cluster assignment = -1)
        cluster_assignments, _ = self.predict_cluster(all_embeddings)
        outlier_indices = np.where(cluster_assignments == -1)[0]
        outlier_embeddings = all_embeddings[outlier_indices]

        if len(outlier_indices) > 0:
            # Split outlier_embeddings into batches
            outlier_batches = np.array_split(outlier_embeddings, len(outlier_embeddings) // batch_size + 1)
            centroids_index = faiss.read_index(f"{index_path}/centroids.index")
            
            for idx, outlier_batch in tqdm(enumerate(outlier_batches), desc="Assigning clusters to outliers"):
                # Using slicing to determine the starting and ending indices for assignment in outlier_indices
                start_idx = idx * len(outlier_batch)
                end_idx = start_idx + len(outlier_batch)
                
                # Get the specific slice of outlier_indices that corresponds to our current batch
                current_outlier_indices = outlier_indices[start_idx:end_idx]
                
                new_assignments = self.find_closest_clusters(centroids_index, outlier_batch, k)
                print(f"New Assignments: {new_assignments}")  # Let's print the new assignments
                
                # Now, use current_outlier_indices to update the actual cluster_assignments
                cluster_assignments[current_outlier_indices] = new_assignments
            
            # After the loop, print any remaining -1s in the cluster_assignments
            print(f"Remaining outliers: {np.where(cluster_assignments == -1)[0]}")



        # Now that all embeddings have a valid cluster, sort by cluster assignments
        sorted_order = np.argsort(cluster_assignments)
        all_embeddings = all_embeddings[sorted_order]
        all_labels = all_labels[sorted_order]
        cluster_assignments = cluster_assignments[sorted_order]

        correct_predictions = 0
        total_predictions = 0

        current_cluster = cluster_assignments[0]
        faiss_index = faiss.read_index(f"{index_path}/prototypes_{current_cluster}.index")
        # Second pass: compute predictions using only the necessary indexes
        for idx, (embedding, label) in  tqdm(enumerate(zip(all_embeddings, all_labels)), desc="Computing predictions during evaluation"):
            if cluster_assignments[idx] != current_cluster:
                # Switch to the next cluster index
                current_cluster = cluster_assignments[idx]
                faiss_index = faiss.read_index(f"{index_path}/prototypes_{current_cluster}.index")

            class_scores, class_indices = self.compute_class_scores({current_cluster: faiss_index}, embedding)
            predicted_class = class_indices[np.argmax(class_scores)]

            correct_predictions += (predicted_class == label.item())
            total_predictions += 1

        accuracy = correct_predictions / total_predictions
        return accuracy

    def get_features(self, dataset, max_length=512, batch_size=16):
        self.model.eval()
        features = []
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for batch in tqdm(dataloader, desc='Get embeddings...'):
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model.base_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
            features.append(cls_embeddings)
        features = np.concatenate(features)
        return features

    def run_clustering(self, embeddings, normalize=False):

        # normalized_embeddings = normalize(cp.array(embeddings))

        if self.umap_model is None:
            self.umap_model = UMAP(**self.umap_config,verbose=True).fit(embeddings)

        umap_embeddings = self.umap_model.transform(embeddings)
        
        if self.cluster_predictor is None:
            self.cluster_predictor = HDBSCAN(**self.hdbscan_config,prediction_data=True,verbose=True).fit(umap_embeddings)
            
        cluster_labels = self.cluster_predictor.labels_
        return cluster_labels

    def predict_cluster(self, embeddings):
        # normalized_embeddings = normalize (cp.array(embeddings))

        if self.umap_model is not None:
            umap_embeddings = self.umap_model.transform(embeddings)
        else:
            raise ValueError("UMAP model is not yet initialized. Run clustering first.")
        
        cluster_labels = approximate_predict(self.cluster_predictor, umap_embeddings)
        return cluster_labels

    # Added saving and loading methods for UMAP and HDBSCAN:

    def save_umap_and_hdbscan_models(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.save_umap_model(path + '/umap_model.joblib')
        self.save_hdbscan_model(path + '/hdbscan_model.joblib')
        # Save the UMAP and HDBSCAN configs as well
        with open(path + '/umap_config.json', 'w') as f:
            json.dump(self.umap_config, f)
        with open(path + '/hdbscan_config.json', 'w') as f:
            json.dump(self.hdbscan_config, f)

    def save_umap_model(self, path: str):
        joblib.dump(self.umap_model, path)

    def save_hdbscan_model(self, path: str):
        joblib.dump(self.cluster_predictor, path)

    def load_umap_model(self, path: str):
        self.umap_model = joblib.load(path)

    def load_hdbscan_model(self, path: str):
        self.cluster_predictor = joblib.load(path)

