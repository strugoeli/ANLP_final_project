from torch.utils.data import Dataset
from collections import Counter
import numpy as np
import torch
import datasets
from clustering import Clustering
from sklearn.preprocessing import LabelEncoder
from transformers import AutoAdapterModel, AutoTokenizer


class FewShotTaskDatasetWithWeights(Dataset):
    def __init__(self, dataset: Dataset, n_way: int, n_support: int, n_query: int):
        # Sorting the dataset based on clusters
        self.sorted_indices = np.argsort(dataset['cluster'])
        self.dataset = dataset

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.clusters, cluster_starts = np.unique(self.dataset['cluster'], return_index=True)

        # Getting end indices for clusters by rolling and slicing
        self.cluster_ends = np.roll(cluster_starts, -1)[:-1]
        self.cluster_ends = np.append(self.cluster_ends, len(self.dataset))

        self.category_labels = np.array(self.dataset['category'])
        self.subcategory_labels = np.array(self.dataset['subcategory'])

        self.cluster_ranges = dict(zip(self.clusters, zip(cluster_starts, self.cluster_ends)))
        self.majority_info, self.class_weights = self.compute_weights_and_majority_categories_subcategories()

    def get_indices_for_cluster(self, cluster):
        start, end = self.cluster_ranges[cluster]
        return self.sorted_indices[start:end]

    def compute_weights_and_majority_categories_subcategories(self, threshold=0.05):
        majority_info = {}
        class_weights = {}

        cluster_labels_np = np.array(self.dataset['cluster'])
        category_labels_np = np.array(self.dataset['category'])
        subcategory_labels_np = np.array(self.dataset['subcategory'])

        for cluster in self.clusters:
            cluster_mask = (cluster_labels_np == cluster)

            # Extract category and subcategory labels for the current cluster
            categories_in_cluster = category_labels_np[cluster_mask]
            subcategories_in_cluster = subcategory_labels_np[cluster_mask]

            # Compute category frequencies
            unique_categories, category_counts = np.unique(categories_in_cluster, return_counts=True)
            total_samples = len(categories_in_cluster)

            # Identify majority categories
            majority_categories = [cat for cat, count in zip(unique_categories, category_counts) if
                                   count / total_samples >= threshold]

            if len(majority_categories) == 0:
                majority_categories = [unique_categories[np.argmax(category_counts)]]

            # For each majority category, list the associated subcategories
            majority_subcategories = {}
            for major_cat in majority_categories:
                associated_subcats = subcategories_in_cluster[categories_in_cluster == major_cat]
                # all associated_subcats with more then self.n_support + self.n_query samples
                associated_subcats = [subcat for subcat in associated_subcats if
                                      np.sum(subcategories_in_cluster == subcat) >= self.n_support + self.n_query]
                majority_subcategories[major_cat] = list(set(associated_subcats))

            majority_info[cluster] = {
                "major_categories": majority_categories,
                "associated_subcategories": majority_subcategories
            }

            # Compute subcategory weights for the current cluster
            subcategory_freq = np.bincount(subcategories_in_cluster)
            weights = np.divide(1., subcategory_freq, out=np.zeros(subcategory_freq.shape), where=subcategory_freq != 0)

            # Normalize the weights so they sum up to 1
            weights /= weights.sum()
            class_weights[cluster] = weights

        return majority_info, class_weights

    def __getitem__(self, index):
        cluster = self.clusters[index]
        cluster_mask = (self.dataset['cluster'] == cluster)

        # Extract major categories and their associated subcategories for this cluster
        major_categories = self.majority_info[cluster]["major_categories"]
        associated_subcategories = self.majority_info[cluster]["associated_subcategories"]

        # Collect all subcategories from major categories
        all_major_subcategories = [subcat for major_cat in major_categories for subcat in
                                   associated_subcategories[major_cat]]
        # Fetch all indices corresponding to this cluster
        cluster_indices = torch.where(cluster_mask)[0].numpy()
        freqnecy = Counter(self.subcategory_labels[cluster_indices])

        # only select the subcategories with more than self.n_support + self.n_query samples
        all_major_subcategories = [subcat for subcat in all_major_subcategories if
                                   freqnecy[subcat] >= self.n_support + self.n_query]
        all_minor_subcategories = [subcat for subcat in freqnecy.keys() if
                                   subcat not in all_major_subcategories and freqnecy[
                                       subcat] >= self.n_support + self.n_query]
        available_subcategories = set(all_major_subcategories + all_minor_subcategories)
        weights = []

        # select n_way subcategories from the major subcategories
        selected_major_subcategories = np.random.choice(all_major_subcategories, min(len(major_categories), self.n_way),
                                                        replace=False)

        # Sample k_shot instances from each majority class for the support set
        support_indices = []
        query_indices = []

        for subcat in selected_major_subcategories:
            indices_of_subcat_in_cluster = cluster_indices[self.subcategory_labels[cluster_indices] == subcat]
            selected_indices = np.random.choice(indices_of_subcat_in_cluster, self.n_support + self.n_query,
                                                replace=False)
            support_indices.extend(selected_indices[:self.n_support])
            query_indices.extend(selected_indices[self.n_support:])
            available_subcategories.remove(subcat)
            weights.extend([self.class_weights[cluster][subcat]] * (self.n_support + self.n_query))

        # select n_way subcategories from all available subcategories
        selected_subcategories = np.random.choice(list(available_subcategories),
                                                  min(len(available_subcategories), self.n_way), replace=False)
        # pick min(len(available_subcategories), self.n_way) subcategories from all available subcategories 
        # and sample 1 instance from each of them for the query set

        for subcat in selected_subcategories:
            indices_of_subcat_in_cluster = cluster_indices[self.subcategory_labels[cluster_indices] == subcat]
            query_indices.extend(selected_indices)
            weights.extend([self.class_weights[cluster][subcat]] * self.n_query)

        support_out = self.dataset[support_indices]
        query_out = self.dataset[query_indices]
        support_out_final = {'input_ids': support_out['input_ids'], 'attention_mask': support_out['attention_mask'],
                             'labels': support_out['subcategory']}
        query_out_final = {'input_ids': query_out['input_ids'], 'attention_mask': query_out['attention_mask'],
                           'labels': query_out['subcategory']}

        return support_out_final, query_out_final, torch.Tensor(weights)

    def __len__(self):
        return len(self.clusters)


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
    adapter_name = encoder.load_adapter("Elise-hf/distilbert-base-uncased_reddit_categories_unipelft", source="hf",
                                        set_active=True)

    # Load the dataset
    dataset = datasets.load_dataset("Elise-hf/reddit_categories_clean")

    # Fit the LabelEncoder on the 'category' column
    subcategory_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    category_encoder.fit(dataset['train']['category'])
    subcategory_encoder.fit(dataset['train']['subcategory'])

    # Encode the dataset
    tokenized_dataset = dataset['validation'].map(encode_batch, batched=True, num_proc=6,
                                                  remove_columns=["text", "category", "subcategory"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "subcategory"])

    # Load the embeddings
    embeddings = np.load('embeddings_fine_tuned/val/val_embeddings.npy')

    # Create the clustering object and run the clustering
    clust = Clustering(encoder, tokenizer)
    cluster_labels = clust.run_clustering(embeddings)
    tokenized_dataset = tokenized_dataset.add_column('cluster', cluster_labels)
    tokenized_dataset = tokenized_dataset.filter(lambda example: example['cluster'] != -1)
    n_way = 2
    n_support = 2
    n_query = 2
    dataset_fw = FewShotTaskDatasetWithWeights(tokenized_dataset, n_way, n_support, n_query)
    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset_fw, batch_size=1)
    print(next(iter(data_loader)))



