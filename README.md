
---

# Reddit Data Preprocessing, Embedding, and Zero-Shot Classification

This repository is dedicated to scripts and models essential for preprocessing Reddit data, generating embeddings with a fine-tuned DistilBERT model equipped with an adapter, and subsequently classifying data into subcategories through a semantic search-based mechanism.

## Getting Started

The project is powered by cuML, a library collection crafted for GPU-accelerated machine learning, complementing RAPIDS projects. Its performance surpasses traditional CPU-driven methods. Ensure you've installed cuML before running the project's code.

For other dependencies:
```sh
pip install -r requirements.txt
```

## Reddit Data Preprocessing

This script has been tailored to clean and preprocess Reddit self-posts.

### Usage
```python
python preprocessing/download_and_clean_dataset.py --dataset_path your_dataset_path --output_dir your_output_dir --output_name your_dataset_name
```

## Reddit Data Embedding & Fine-tuning

The `EDA_and_Training.ipynb` notebook delineates the fine-tuning process of a pretrained DistilBERT model, augmented with an adapter for the "category" label. Subsequent to this, the script computes embeddings for Reddit's self-posts.

### Usage
```python
python preprocessing/create_embeddings.py --adapter_path your_adapter_path --output_dir your_output_dir
```

## Embedding Clusterer 

Post embedding, the data undergoes dimensionality reduction via UMAP and clustering with KMeans.

### Usage

Replace `your_embeddings`:
```python
embeddings = your_embeddings  
embedding_clusterer = EmbeddingClusterer()
clusters = embedding_clusterer.run_clustering(embeddings)
```

For model persistence:

```python
# Save models
embedding_clusterer.save_umap_and_cluster_predictor('path_to_save_models')

# Load models
embedding_clusterer.load_umap_model('path_to_models')
embedding_clusterer.load_cluster_predictor_model('path_to_models')
```

## Zero-Shot Classification with Semantic Search

The main classification method is presented in `inference.py`. After tagging test points to clusters, a semantic search is employed to classify them into "subcategories." This search is confined to predicted clusters, with train set embeddings indexed for efficiency. Majority voting, influenced by the nearest train set neighbors, determines the subcategory for the test point.

### Usage
```sh
python inference.py --train_samples_ratio 0.25 --seed 42 --model_path "data/ds_val_clusters_info_kmeans_80"
```

---

