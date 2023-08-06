import pandas as pd
from transformers import AutoAdapterModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import datasets
from cluster_based_classifier import ClusterBasedClassifier
from search_utils import compute_and_store_prototypes_centroids, create_faiss_indexes
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from search_utils import get_representative
from tqdm import tqdm
# tokenized_dataset = dataset.map(encode_batch, batched=True, num_proc=6, remove_columns=["text", "category", "subcategory",])

# tokenized_dataset = tokenized_dataset.remove_columns(['__index_level_0__', 'text_word_count', 'title_word_count'])
# encoder.to(device)
# encoder.eval()

# tokenized_dataset.set_format( type="torch", columns=["input_ids", "attention_mask"] ,device=device)
# tokenized_dataset = tokenized_dataset.map(batch_embeddings, batched=True,batch_size=128, desc=' Computing embeddings')
# tokenized_dataset.reset_format()
# tokenized_dataset.save_to_disk("data/reddit_categories_clean_embeddings",num_proc=6)
# print(tokenized_dataset)

#
# def batch_embeddings(batch):
#     with torch.no_grad():
#         # Move the batch to the device
#
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         # Get the embeddings
#         embeddings = encoder.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
#
#         return {"embeddings": embeddings.detach().cpu().numpy()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_nearest_neighbors(batch):
    """Computes the nearest neighbors of a set of embeddings."""
    scores, retrieved_examples = ds_train.get_nearest_examples_batch('embeddings', batch['embeddings'], k=5)
    preds_subcategory = [x['subcategory'] for x in retrieved_examples]
    preds_subcategory_mv = [Counter(x).most_common(1)[0][0] for x in preds_subcategory]

    return {"preds_subcategory": preds_subcategory,'scores':scores,'max_voting':preds_subcategory_mv}

ADAPTER_PATH = "Elise-hf/distilbert-base-uncased_reddit_categories_unipelft"
if __name__ == '__main__':
    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        # Concatenate title and selftext with [SEP] token in between
        combined_text = [title + ' [SEP] ' + selftext for title, selftext in zip(batch['title'], batch['text'])]

        # Encode the text using the tokenizer and add the attention mask
        encoding = tokenizer(combined_text, max_length=512, truncation=True, padding="longest")
        # Encode the labels and add them to the encoding
        encoding['subcategory'] = subcategory_encoder.transform(batch['subcategory'])
        encoding['category'] = category_encoder.transform(batch['category'])
        return encoding


    # Load the model and tokenizer
    encoder = AutoAdapterModel.from_pretrained("distilbert-base-uncased",cache_dir='model_weights')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Load the adapter
    adapter_name = encoder.load_adapter(ADAPTER_PATH, source="hf",set_active=True)

    # Load the dataset
    dataset = datasets.load_dataset("data/reddit_categories_clean_embeddings",name="reddit_categories_clean_embeddings")
    # Fit the LabelEncoder on the 'category' column
    subcategory_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    category_encoder.fit(dataset['train']['category'])
    subcategory_encoder.fit(dataset['train']['subcategory'])

    # dataset['train'].add_faiss_index('embeddings',device=0)
    ds_train = dataset['train']
    ds_val = dataset['validation']
    val_embed = np.array(ds_val['embeddings'])
    train_embed = np.array(ds_train['embeddings'])

    # scores, retrieved_examples =ds_train.get_nearest_examples_batch('embeddings', np.array(ds_val['embeddings']), k=5)

    # Create the clustering object and run the clustering
    cluster_based_classifier = ClusterBasedClassifier(encoder, tokenizer)
    cluster_labels = cluster_based_classifier.run_clustering(val_embed)

    cluster_based_classifier.save_umap_and_hdbscan_models('clustering_model')
    ds_val = ds_val.add_column('cluster', cluster_labels)
    ds_val_filtered = ds_val.filter(lambda example: example['cluster'] != -1)

    filtered_clusters = np.array(ds_val_filtered['cluster'])
    filtered_embed = np.array(ds_val_filtered['embeddings'])

    unique_clusters = np.unique(filtered_clusters)
    clusters_representatives = []
    for cluster_id in tqdm (unique_clusters, desc='Computing cluster representatives'):
        cluster_embed = filtered_embed[filtered_clusters==cluster_id]
        mean_representative = get_representative(cluster_embed, 'mean')
        centroid_representative = get_representative(cluster_embed, 'centroid')
        row = { 'cluster':cluster_id,
                'mean_representative':mean_representative,
                'centroid_representative':centroid_representative,
                'num_examples':len(cluster_embed),
                }
        clusters_representatives.append(row)

    ds_train_clusters =datasets.Dataset.from_list(clusters_representatives)
    ds_train_clusters.save_to_disk('data/ds_val_clusters_info2')

    c=0



    #
    # tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "cluster"])
    # compute_and_store_prototypes_centroids(tokenized_dataset, encoder, out_dir='embeddings')
    #
    # create_faiss_indexes('embeddings', 'embeddings_fine_tuned/val/val_indexes')
    # tokenized_dataset_test = dataset['test'].map(encode_batch, batched=True, num_proc=6,
    #                                              remove_columns=["text", "category", "subcategory"])
    # tokenized_dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "subcategory"])
    # small_test = tokenized_dataset_test.select(range(100))
    # test_loader = DataLoader(small_test, batch_size=32, shuffle=False)
    #
    # index_path = "embeddings_fine_tuned/val/val_indexes"
    # res = cluster_based_classifier.evaluate(test_loader, index_path)
    # print('Accuracy: ', res)
