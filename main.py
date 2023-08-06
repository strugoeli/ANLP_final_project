
from transformers import AutoAdapterModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import datasets
from cluster_based_classifier import ClusterBasedClassifier
from search_utils import compute_and_store_prototypes_centroids, create_faiss_indexes
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch




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
    encoder = AutoAdapterModel.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Load the adapter
    adapter_name = encoder.load_adapter("Elise-hf/distilbert-base-uncased_reddit_categories_unipelft", source="hf", set_active=True)

    # Load the dataset
    dataset = datasets.load_dataset("Elise-hf/reddit_categories_clean")

    # Fit the LabelEncoder on the 'category' column
    subcategory_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    category_encoder.fit(dataset['train']['category'])
    subcategory_encoder.fit(dataset['train']['subcategory'])

    # Encode the dataset
    tokenized_dataset = dataset['validation'].map(encode_batch, batched=True,num_proc=6,remove_columns=["text","category","subcategory"])

    # Load the embeddings
    embeddings = np.load('embeddings_fine_tuned/val/val_embeddings.npy')

    # Create the clustering object and run the clustering
    cluster_based_classifier = ClusterBasedClassifier(encoder,tokenizer)
    cluster_labels = cluster_based_classifier.run_clustering(embeddings)
    cluster_based_classifier.save_umap_and_hdbscan_models('clustering_model')

    tokenized_dataset = tokenized_dataset.add_column('cluster', cluster_labels)
    tokenized_dataset = tokenized_dataset.filter(lambda example: example['cluster'] != -1)

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "cluster"])


    compute_and_store_prototypes_centroids(tokenized_dataset, encoder, out_dir='embeddings')
    create_faiss_indexes('embeddings', 'embeddings_fine_tuned/val/val_indexes')
    tokenized_dataset_test = dataset['test'].map(encode_batch, batched=True,num_proc=6,remove_columns=["text","category","subcategory"])
    tokenized_dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "subcategory"])
    small_test = tokenized_dataset_test.select(range(100))
    test_loader = torch.utils.data.DataLoader(small_test, batch_size=32, shuffle=False)

    index_path = "embeddings_fine_tuned/val/val_indexes"
    res = cluster_based_classifier.evaluate(test_loader, index_path)
    print('Accuracy: ', res)