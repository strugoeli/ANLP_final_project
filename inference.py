from transformers import AutoAdapterModel, AutoTokenizer
import datasets
from cluster_based_classifier import ClusterBasedClassifier
import numpy as np
from collections import Counter
from tqdm import tqdm
import gc
import faiss
import logging
from collections import defaultdict
import operator
gc.collect()
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix

ADAPTER_PATH = "Elise-hf/distilbert-base-uncased_reddit_categories_unipelft"

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



def get_pred(scores, examples , weighted=False):
     if weighted:
        votes = defaultdict(float)
        weights = 1 / (scores + 1e-5)
        # Calculate weighted votes
        for weight, example in zip(weights, examples):
         votes[example['subcategory']] += weight
         # Determine predicted label
        pred = max(votes.items(), key=operator.itemgetter(1))[0]
        return pred

     preds_majority = [Counter(x).most_common(1)[0][0] for x in raw_preds]
     return  preds_majority

def get_cluster_recall_and_density(ds_train, ds_test):

    # Convert to pandas dataframes
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



if __name__ == '__main__':
    logger = init_logger()

    # Load the model and tokenizer
    logger.info("Loading model and tokenizer")
    encoder = AutoAdapterModel.from_pretrained("distilbert-base-uncased", cache_dir='model_weights')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Load the adapter
    adapter_name = encoder.load_adapter(ADAPTER_PATH, source="hf", set_active=True)

    # Load the dataset
    logger.info("Loading dataset")
    dataset = datasets.load_dataset("data/reddit_categories_clean_embeddings",
                                    name="reddit_categories_clean_embeddings")
    num_sample = len(dataset['train']) // 4
    ds_train = dataset['train'].shuffle(seed=42).select(range(num_sample))
    # ds_val = dataset['validation']
    ds_test = dataset['test']

    logger.info("Adding faiss index")
    logger.info("Setting format")
    dataset.set_format(type='numpy', columns=['category', 'subcategory', 'embeddings'])
    # val_embed = np.array(ds_val['embeddings'])
    test_embed = np.array(ds_test['embeddings'])
    train_embed = np.array(ds_train['embeddings'])

    logger.info("Loading cluster based classifier")
    cluster_based_classifier = ClusterBasedClassifier(encoder, tokenizer)

    # cluster_predictor = cluster_based_classifier.load_cluster_predictor_model('data/ds_val_clusters_info_kmeans')
    # umap_model = cluster_based_classifier.load_umap_model('data/ds_val_clusters_info_kmeans')

    logger.info("Train cluster predictor model on train set")
    cluster_labels_train = cluster_based_classifier.run_clustering(train_embed)
    centroids = cluster_based_classifier.get_centroids()
    #
    # centroids_index = faiss.IndexFlatL2(centroids.shape[1])
    # centroids_index.add(centroids.astype(np.float32))
    # _, centroids_labels = centroids_index.search(test_embed.astype(np.float32), 2)

    logger.info("Saving cluster predictor model and umap model")
    cluster_based_classifier.save_umap_and_cluster_predictor("data/ds_val_clusters_info_kmeans_80")

    logger.info("Predicting clusters for test set")
    cluster_labels = cluster_based_classifier.predict_cluster(test_embed)

    logger.info("Adding cluster column to train set and test set")
    ds_test = ds_test.add_column('cluster', cluster_labels)
    ds_train = ds_train.add_column('cluster', cluster_labels_train)

    logger.info("Checking how many time the subcategory of the test set appears in each cluster predicted by the cluster predictor")
    df_test = get_cluster_recall_and_density(ds_train,ds_test)

    logger.info("Adding subcategory column to train set and test set")
    ds_train_per_cluster = {}
    for cluster_id in np.unique(cluster_labels_train):
        index_cluster = np.where(cluster_labels_train == cluster_id)[0]
        current_ds = ds_train.select(index_cluster)
        current_ds.add_faiss_index('embeddings')
        ds_train_per_cluster[cluster_id] = current_ds

    logger.info("Adding subcategory column to train set and test set")
    cluster_assignments = np.array(ds_test['cluster'])
    all_embeddings = np.array(ds_test['embeddings'])
    cluster_labels = np.array(ds_test['subcategory'])

    # Get the class scores and indices for each cluster
    logger.info("Getting class scores and indices for each cluster")
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
            current_preds =get_pred( examples=raw_preds, scores=class_scores, weighted=True)
            correct_predictions += np.sum(current_preds == batch_current_labels)
            total_predictions += len(current_preds)
            predictions.extend(current_preds)
            ground_truth.extend(batch_current_labels)



    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    logger.info("Accuracy: {}".format(accuracy_score(ground_truth, predictions)))
    logger.info("Precision: {}".format(precision_score(ground_truth, predictions, average='macro')))
    logger.info("Recall: {}".format(recall_score(ground_truth, predictions, average='macro')))
    logger.info("F1: {}".format(f1_score(ground_truth, predictions, average='macro')))
    # logger.info("Classification report: {}".format(classification_report(ground_truth, predictions)))
    # logger.info("Confusion matrix: {}".format(confusion_matrix(ground_truth, predictions)))

# conda create --name anlp_r python=3.10
