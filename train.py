
from transformers import AutoAdapterModel, AutoTokenizer
import datasets
from sklearn.preprocessing import LabelEncoder
import numpy as np
from clustering import Clustering
from prototypical_net import PrototypicalNetworks
import torch
from torch.utils.data import DataLoader


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


encoder = AutoAdapterModel.from_pretrained("distilbert-base-uncased")
adapter_name = encoder.load_adapter("Elise-hf/distilbert-base-uncased_reddit_categories_unipelft", source="hf", set_active=True)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
dataset = datasets.load_dataset("Elise-hf/reddit_categories_clean")


# Fit the LabelEncoder on the 'category' column
subcategory_encoder = LabelEncoder()
category_encoder = LabelEncoder()
category_encoder.fit(dataset['train']['category'])
subcategory_encoder.fit(dataset ['train']['subcategory'])



tokenized_dataset = dataset['validation'].map(encode_batch, batched=True, num_proc=6,
                                              remove_columns=["text", "category", "subcategory"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "subcategory"])

dataset = dataset.select_columns(['title', 'text', 'category', 'subcategory'])




tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "subcategory"])

np.save('embeddings_fine_tuned/val/val_embeddings.npy', emmbeddings)

clust = Clustering(encoder, tokenizer)

emmbeddings = np.load('embeddings_pre_trained/val/val_embeddings.npy')
cluster_labels = clust.run_clustering(emmbeddings)




tokenized_dataset = tokenized_dataset.add_column('cluster', cluster_labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PrototypicalNetworks(encoder).to(device)

ds_val = dataset['validation']
ds_val = ds_val.add_column('cluster_labels', cluster_labels)
tokenized_dataset.cast_column('cluster', np.array)




tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'cluster', 'category', 'subcategory'])
few_shot_dataset = FewShotTaskDatasetWithWeights(tokenized_dataset, n_way=5, n_support=5, n_query=5)
few_shot_dataloader = DataLoader(few_shot_dataset, batch_size=1)









dataset_fw = FewShotTaskDataset(tokenized_dataset, major_classes, class_weights)

# %%
cluster = 0
majority_classes = dataset_fw.major_classes[cluster]

# Get the indices of the data points in this cluster
cluster_indices = np.where(dataset_fw.cluster_labels == cluster)[0]

# Get the category labels for this cluster
cluster_categories = dataset_fw.category_labels[cluster_indices]

# Find the indices of the majority and minority class instances
majority_indices = [i for i, category in enumerate(cluster_categories) if category in majority_classes]
minority_indices = [i for i, category in enumerate(cluster_categories) if category not in majority_classes]

# Sample k_shot instances from each majority class for the support set
support_set = []
for class_ in majority_classes:
    # class_indices = [i for i in majority_indices if cluster_categories[i] == class_]
    class_indices = cluster_categories[cluster_categories == class_].index.tolist()
    support_set += list(np.random.choice(class_indices, dataset_fw.k_shot, replace=False))

# Sample the remaining instances for the support set from the minority classes
if len(support_set) < dataset_fw.k_shot * len(majority_classes):
    support_set += list(
        np.random.choice(minority_indices, dataset_fw.k_shot * len(majority_classes) - len(support_set), replace=False))

# Sample n_query instances for the query set
remaining_indices = list(set(cluster_indices) - set(support_set))
query_set = list(np.random.choice(remaining_indices, dataset_fw.n_query, replace=False))

# Get the class weights for this cluster
weights = dataset_fw.class_weights[cluster]



cluster_categories[cluster_categories == class_].index.tolist()

from torch.utils.data import DataLoader

data_loader = DataLoader(dataset_fw, batch_size=1)




