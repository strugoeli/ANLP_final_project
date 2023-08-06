from scipy.spatial.distance import cdist
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_representative(embeddings, representative_type="mean"):
    """
    Compute the representative point of a set of embeddings.

    Args:
    embeddings (np.array): The embeddings of the instances in a cluster.
    representative_type (str): The type of representative to compute. Options are 'mean', 'centroid', and 'medoid'.

    Returns:
    representative (np.array): The representative point of the embeddings.
    """
    if representative_type == "mean":
        representative = np.mean(embeddings, axis=0)
    elif representative_type == "centroid":
        representative = embeddings[np.argmin(cdist(embeddings, embeddings).sum(axis=1))]
    elif representative_type == "medoid":
        representative = embeddings[np.argmin(cdist(embeddings, embeddings).mean(axis=1))]
    else:
        raise ValueError(f"Unknown representative type: {representative_type}. Choices are 'mean', 'centroid', and 'medoid'.")

    return representative

def compute_and_store_prototypes_centroids(ds, model,out_dir, representative_type="centroid",embedding_path=None):
    model.eval()
    model.to(device)
    prototypes = defaultdict(list)
    os.makedirs(out_dir, exist_ok=True)

    if embedding_path is not None:

        with open(os. path.join(embedding_path, "all_embeddings.pkl"), "rb") as f:
            all_embeddings = pickle.load(f)
        with open(os. path.join(embedding_path, "all_clusters.pkl"), "rb") as f:
            all_clusters = pickle.load(f)
        with open(os. path.join(embedding_path, "all_subcategories.pkl"), "rb") as f:
            all_subcategories = pickle.load(f)

        # Convert lists to numpy arrays for efficient computations
        unique_clusters = np.unique(all_clusters)

    else:

        train_loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

        all_embeddings = []
        all_clusters = []
        all_subcategories = []

        # Step 1: Computing the embeddings for the entire train dataset
        with torch.no_grad():
            for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy() 
                all_embeddings.extend(embeddings)

        # Convert lists to numpy arrays for efficient computations
        all_embeddings = np.array(all_embeddings)
        all_clusters = np.array(ds['cluster'])
        all_subcategories = np.array(ds['subcategory'])
        unique_clusters = np.unique(all_clusters)

        with open(os.path.join(out_dir, "all_embeddings.pkl"), "wb") as f:
            pickle.dump(all_embeddings, f)
        
        # save cluster and subcategory information
        with open(os.path.join(out_dir, "all_clusters.pkl"), "wb") as f:
            pickle.dump(all_clusters, f)

        with open(os.path.join(out_dir, "all_subcategories.pkl"), "wb") as f:
            pickle.dump(all_subcategories, f)

    # Step 2: Computing the prototype (mean embedding) for each class within each cluster
    for cluster in unique_clusters:
        cluster_mask = all_clusters == cluster
        unique_subcategories_in_cluster = np.unique(all_subcategories[cluster_mask])

        for subcategory in unique_subcategories_in_cluster:
            subcategory_mask = all_subcategories == subcategory
            mask = cluster_mask & subcategory_mask

            class_embeddings = all_embeddings[mask]
            prototype = get_representative(class_embeddings, representative_type) 
            prototypes[cluster].append((subcategory, prototype))

    with open(os.path.join(out_dir, "prototypes.pkl"), "wb") as f:
        pickle.dump(prototypes, f)

    # Step 3: Computing the centroid for each cluster
    centroids = {}
    for cluster in unique_clusters:
        cluster_mask = all_clusters == cluster
        embeddings = all_embeddings[cluster_mask]
        centroid = get_representative(embeddings, representative_type)
        centroids[cluster] = centroid
    
    # Storing results
    with open(os.path.join(out_dir, "centroids.pkl"), "wb") as f:
        pickle.dump(centroids, f)


def create_faiss_indexes(input_path, index_path):
    # Load the prototypes, centroids, and labels
    prototype_path = os.path.join(input_path, "prototypes.pkl")
    centroid_path = os.path.join(input_path, "centroids.pkl")
    all_labels_path = os.path.join(input_path, "all_subcategories.pkl")
    
    with open(prototype_path, "rb") as f:
        prototypes = pickle.load(f)
    with open(centroid_path, "rb") as f:
        centroids = pickle.load(f)
    with open(all_labels_path, "rb") as f:
        all_labels = pickle.load(f)

    os.makedirs(index_path, exist_ok=True)

    prototype_indexes = {}

    # Consolidated Centroid Index
    centroid_dimension = list(centroids.values())[0].shape[0]
    consolidated_centroid_index = faiss.IndexFlatL2(centroid_dimension)
    centroids_matrix = np.vstack(list(centroids.values()))
    consolidated_centroid_index.add(centroids_matrix)

    for cluster, cluster_prototypes in prototypes.items():
        # Create a base Faiss index for the prototypes of this cluster
        d = cluster_prototypes[0][1].shape[0]
        index_base = faiss.IndexFlatL2(d)
        
        # Wrap with IDMap to store labels alongside vectors
        index_with_ids = faiss.IndexIDMap2(index_base)
        
        # Extract labels and embeddings for this cluster's prototypes
        cluster_labels = np.array([all_labels[proto[0]] for proto in cluster_prototypes]).astype(np.int64)
        cluster_embeddings = np.vstack([proto[1] for proto in cluster_prototypes])
        
        # Add prototypes and their labels to the index
        index_with_ids.add_with_ids(cluster_embeddings, cluster_labels)
        prototype_indexes[cluster] = index_with_ids

    # Save the indexes using Faiss's built-in functions
    for cluster, index in prototype_indexes.items():
        faiss.write_index(index, f"{index_path}/prototypes_{cluster}.index")
    
    # Save consolidated centroid index
    faiss.write_index(consolidated_centroid_index, f"{index_path}/centroids.index")



