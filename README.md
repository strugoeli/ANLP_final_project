
# Reddit Data Preprocessing and Classification 

This repository contains scripts for preprocessing Reddit data, generating embeddings using a pre-trained transformer model with an adapter, and a Cluster-Based Classifier (CBC) for classification of the preprocessed and embedded data. 

## Getting Started

This project uses cuML, a suite of libraries for GPU-accelerated machine learning that's compatible with RAPIDS projects. It provides faster operations than traditional CPU-based methods.
Installation details and complete info available in the official documentation. [https://docs.rapids.ai/install ]
Please install cuML before running our code.



Please install the required dependencies listed in the `requirements.txt` file. Use the following command to install these dependencies:

```sh
pip install -r requirements.txt
```

## Reddit Data Preprocessing

This script cleans and preprocesses Reddit self-posts. 

### Usage

Run the script from the command line with optional arguments:

- `--dataset_path`: Alternative dataset path. Default: `Elise-hf/reddit-self-post`.
- `--output_dir`: Directory to save the cleaned dataset. Default: `output`.
- `--output_name`: Name for the cleaned dataset. Default: `reddit_data_cleaned`.

Example:
```
python preprocessing/download_and_clean_dataset.py --dataset_path your_dataset_path --output_dir your_output_dir --output_name your_dataset_name
```

## Reddit Data Embedding

This script computes embeddings for Reddit self-posts using a pre-trained Transformer model and an adapter trained on the Reddit dataset.

### Usage

Run the script from the command line with optional arguments:

- `--adapter_path`: Path to the adapter weights. 
- `--batch_size`: Batch size for computing embeddings.
- `--num_proc`: Number of processes to use for saving embeddings. 
- `--output_dir`: Directory to save embeddings. 
- `--set_active`: Set the adapter as active. Default: `True`.
- `--split`: Splits to use for creating embeddings. If none, all splits are used. Example: `--split train test`.

Example:
```
python preprocessing/create_embeddings.py --adapter_path your_adapter_path --output_dir your_output_dir
```

## Embedding Clusterer

This class applies dimensionality reduction and clustering on embeddings using UMAP and KMeans models. The embeddings are first normalized, then processed with UMAP for dimensionality reduction, and subsequently clustered using KMeans. The UMAP and KMeans models can be saved and loaded for future use.

## Usage

To run the script, replace `your_embeddings` in the following example:

```python
embeddings = your_embeddings  # Replace with your actual embeddings
embedding_clusterer = EmbeddingClusterer()
clusters = embedding_clusterer.run_clustering(embeddings)
```

The `run_clustering` method runs the clustering on the embeddings and returns the cluster labels for each embedding. The `predict_cluster` method predicts the cluster labels for given embeddings based on the previously trained UMAP and KMeans model. 

Model saving and loading can be performed with the following methods:

```python
# Save models
embedding_clusterer.save_umap_and_cluster_predictor('path_to_save_models')

# Load models
embedding_clusterer.load_umap_model('path_to_models')
embedding_clusterer.load_cluster_predictor_model('path_to_models')
```

Replace `'path_to_save_models'` and `'path_to_models'` with your actual paths. 



## Cluster-Based Classifier (CBC)

This script trains a CBC on the `reddit_categories_clean_embeddings` dataset obtained from the Hugging Face datasets library. The CBC model uses a UMAP transformer for dimensionality reduction followed by clustering of the transformed data. 

### Usage

To run the script, execute the following command:

```sh
python inference.py --train_samples_ratio 0.25 --seed 42 --model_path "data/ds_val_clusters_info_kmeans_80"
```

### Arguments

The script accepts the following command-line arguments:

- `train_samples_ratio`: Ratio of training samples to use. Default is 0.25.
- `seed`: Seed for reproducibility. Default is 42.
- `model_path`: Path to save/load the model. Default is "data/ds_val_clusters_info_kmeans_80".




