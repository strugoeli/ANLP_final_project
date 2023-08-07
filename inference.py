import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from datasets import load_dataset
import logging
from clusterer import EmbeddingClusterer
from collections import  defaultdict, Counter
import operator
import argparse

def init_logger(log_name=None):
    # create logger with INFO level
    logger_name = log_name if log_name else __name__
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # if the logger has handlers, it's already been initialized
    if not logger.handlers:
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)
    return logger


def get_pred(scores, examples, weighted=False):
    """
    Computes the predictions based on the highest scores or highest weighted scores.

    Parameters:
        scores (list): A list of scores associated with each instance in the examples.
        examples (list): A list of examples each containing associated classes.
        weighted (bool, optional): A boolean flag to decide if the majority voting should be weighted. Defaults to False.

    Returns:
        list: A list of predicted classes. If `weighted` is set to True, the function returns predictions
              based on weighted majority voting, where weights are calculated as the inverse of the scores
              (with a small offset added for stability). If `weighted` is set to False, the function returns
              predictions based on simple majority voting.

    Note:
        The function uses `operator.itemgetter(1)` to resolve ties in max votes, hence the selection might be arbitrary
        among classes with the same max votes. Consider using a tie-breaking rule for more consistent results.
    """
    if weighted:
        scores = np.array(scores)
        weighted_majority_preds = []
        weights = 1 / (scores + 1e-5)
        # Calculate weighted votes
        for weight, example in zip(weights, examples):
            votes = defaultdict(float)
            for w, label in zip(weight, example):
                votes[label] += w
            pred = max(votes.items(), key=operator.itemgetter(1))[0]
            weighted_majority_preds.append(pred)
        return weighted_majority_preds

    preds_majority = [Counter(x).most_common(1)[0][0] for x in examples]
    return preds_majority

def get_cluster_recall_and_density(ds_train, ds_test):
    """
     This function computes the recall and density of each cluster with respect to subcategories.
     The recall is the proportion of instances of a given subcategory that are assigned to the correct cluster.
     The density is the proportion of instances in a cluster that belong to the most common subcategory.

     Parameters:
     ds_train (Dataset): The training dataset.
     ds_test (Dataset): The test dataset.

     Returns:
     df_test (Pandas DataFrame): A dataframe that includes columns indicating whether the true subcategory
                                 is in the cluster's subcategories (hit),
                                 frequency of each subcategory within each cluster (counts),
                                 the size of each cluster (cluster_size),
                                 and the proportion of each subcategory within its cluster (proportion).
     """

    df_train = ds_train.select_columns(['category', 'cluster', 'subcategory']).to_pandas()
    df_test = ds_test.select_columns(['category', 'cluster', 'subcategory']).to_pandas()

    # Compute the set of subcategories for each cluster in the training data and merge with test data
    subcategories_per_cluster = df_train.groupby('cluster')['subcategory'].apply(set).reset_index(name='subcategories')
    df_test = df_test.merge(subcategories_per_cluster, on='cluster', how='left')

    # Compute 'hit' column which indicates if the true subcategory is in the cluster's subcategories
    df_test['hit'] = df_test.apply(lambda row: row['subcategory'] in row['subcategories'], axis=1)

    # Compute frequency of each subcategory within each cluster in the training data
    subcategory_counts = df_train.groupby(['cluster', 'subcategory']).size().reset_index(name='counts')
    df_test = df_test.merge(subcategory_counts, on=['cluster', 'subcategory'], how='left')
    df_test['counts'] = df_test['counts'].fillna(0)

    # Merge this with test data and compute the proportion of each subcategory within its cluster
    cluster_sizes = df_train['cluster'].value_counts().reset_index()
    cluster_sizes.columns = ['cluster', 'cluster_size']
    df_test = df_test.merge(cluster_sizes, on='cluster')
    df_test['proportion'] = df_test['counts'] / df_test['cluster_size']

    return df_test

def load_datasets():
    logger.info("Loading dataset")
    dataset = load_dataset("Elise-hf/reddit_categories_clean_embeddings")
    return dataset


def create_train_clusters(cluster_labels_train, ds_train):
    """
    This function creates a dictionary of train datasets, where each key is a unique cluster id
    and the value is the corresponding subset of the train dataset associated with that cluster.

    Parameters:
    cluster_labels_train (array): The array of cluster labels for the training data.
    ds_train (Dataset): The training dataset.

    Returns:
    ds_train_per_cluster (dict): A dictionary with cluster ids as keys and corresponding datasets as values.
    """
    ds_train_per_cluster = {}
    for cluster_id in np.unique(cluster_labels_train):
        index_cluster = np.where(cluster_labels_train == cluster_id)[0]
        current_ds = ds_train.select(index_cluster)
        current_ds.add_faiss_index('embeddings')
        ds_train_per_cluster[cluster_id] = current_ds
    return ds_train_per_cluster


def get_class_scores_and_indices(ds_train_per_cluster,ds_test ):
    """
   This function processes the train and test datasets by assigning them into clusters,
   computing the nearest neighbors and predicting the classes based on the clusters.

   Parameters:
   ds_train (Dataset): The training dataset.
   ds_test (Dataset): The test dataset.

   Returns:
   predictions (numpy array): The array of predicted classes.
   ground_truth (numpy array): The array of true classes.
    """

    cluster_assignments = np.array(ds_test['cluster'])
    all_embeddings = np.array(ds_test['embeddings'])
    cluster_labels = np.array(ds_test['subcategory'])

    correct_predictions = 0
    total_predictions = 0
    bz = 128
    predictions = []
    ground_truth = []
    for cluster_id in np.unique(cluster_assignments):
        cluster_embeddings = all_embeddings[cluster_assignments == cluster_id]
        current_ds_train = ds_train_per_cluster[cluster_id]
        current_labels = cluster_labels[cluster_assignments == cluster_id]

        for i in tqdm(range(0, len(cluster_embeddings), bz), desc=f"Cluster {cluster_id}"):
            class_scores, retrieved_examples = current_ds_train.get_nearest_examples_batch('embeddings',
                                                                                           cluster_embeddings[i:i + bz],
                                                                                           k=5)
            raw_preds = [x['subcategory'] for x in retrieved_examples]
            batch_current_labels = current_labels[i:i + bz]
            current_preds = get_pred(examples=raw_preds, scores=class_scores, weighted=True)
            correct_predictions += np.sum(current_preds == batch_current_labels)
            total_predictions += len(current_preds)
            predictions.extend(current_preds)
            ground_truth.extend(batch_current_labels)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    return predictions, ground_truth


def evaluate_model(predictions, ground_truth):
    """
    This function evaluates the model by computing the accuracy, precision, recall and f1 score.
    """
    logger.info("Accuracy: {}".format(accuracy_score(ground_truth, predictions)))
    logger.info("Precision: {}".format(precision_score(ground_truth, predictions, average='macro')))
    logger.info("Recall: {}".format(recall_score(ground_truth, predictions, average='macro')))
    logger.info("F1: {}".format(f1_score(ground_truth, predictions, average='macro')))


def main():
    # Load the dataset
    dataset = load_datasets()

    # Define Cluster Based Classifier
    embedding_clusterer = EmbeddingClusterer()

    # Format the dataset
    ds_test = dataset['test'].rename_column('subcategory', 'labels')
    ds_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Shuffle and select
    num_sample = len(dataset['train']) * args.train_samples_ratio
    ds_train = dataset['train'].shuffle(seed=args.seed).select(range(num_sample))
    ds_test = dataset['test']

    logger.info("Setting format")
    dataset.set_format(type='numpy', columns=['category', 'subcategory', 'embeddings_pretrained'])
    test_embed = np.array(ds_test['embeddings'])
    train_embed = np.array(ds_train['embeddings'])

    logger.info("Train cluster predictor model on train set")
    cluster_labels_train = embedding_clusterer.run_clustering(train_embed)
    # centroids = embedding_clusterer.get_centroids()

    logger.info("Saving cluster predictor model and umap model")
    embedding_clusterer.save_umap_and_cluster_predictor(args.model_path)

    logger.info("Predicting clusters for test set")
    cluster_labels = embedding_clusterer.predict_cluster(test_embed)

    # Add cluster column
    logger.info("Adding cluster column to train set and test set")
    ds_test = ds_test.add_column('cluster', cluster_labels)
    ds_train = ds_train.add_column('cluster', cluster_labels_train)
    ds_train_per_cluster = create_train_clusters(cluster_labels_train, ds_train)

    # Get class scores and indices
    predictions, ground_truth = get_class_scores_and_indices(ds_test=ds_test, ds_train_per_cluster =ds_train_per_cluster)

    # Evaluate model
    evaluate_model(predictions, ground_truth)

if __name__ == '__main__':
    logger = init_logger()
    # Argument parser
    parser = argparse.ArgumentParser(description="Cluster-Based Classifier")
    parser.add_argument("--train_samples_ratio", default=0.25, type=float, help="Ratio of training samples to use")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--model_path", default="data/ds_val_clusters_info_kmeans_80", type=str,
                        help="Path to save/load the model")

    args = parser.parse_args()

    main()
