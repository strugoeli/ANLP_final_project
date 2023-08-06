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
import datasets
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

def get_prototypes(batch):
    pass
embeddings_ds =datasets.load("data/reddit_categories_clean_embeddings")

embeddings = embeddings_ds.map (lambda x: x["embeddings"])





