import os
import numpy as np
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from textacy import extract
import pandas as pd
# import plotly.express as px
import datasets
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("os.environ['LD_LIBRARY_PATH']:", os.environ['LD_LIBRARY_PATH'])
# Now you can import the CUDA libraries and run your code that requires CUDA.
class Clustering:
    def __init__(self, model,tokenizer):
        self.model = model.to(device)
        self.tokenizer = tokenizer


    def get_features(self, dataset, max_length=512, batch_size=16):
        self.model.eval()
        features = []

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for batch in tqdm(dataloader,desc='Get embeddings...'):
            inputs = {k:v.to(device) for k,v in batch.items()}
            outputs = self.model.base_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()  # Use the hidden state of the [CLS] token
            features.append(cls_embeddings)

        features = np.concatenate(features)
        return features
    def run_clustering(self, embeddings, min_cluster_size=30, min_samples=None, n_neighbors=15, n_components=5):
        # Apply UMAP dimensionality reduction
        umap_embeddings = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine').fit_transform(embeddings)
        # Perform HDBSCAN clustering
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
        return clusterer.labels_
    
    def get_top_words(self, texts, n=5):
        top_words = []
        for text in texts:
            words = extract.words(text, filter_stops=True, filter_punct=True, filter_nums=True)
            top_words.append(' | '.join(words[:n]))
        return top_words

    def get_frequencies(self, labels):
        return pd.Series(labels).value_counts().tolist()

    def visualize_clusters(self, embeddings, labels, words_per_topic, frequencies):
        # Reduce dimensionality for visualization
        reducer = UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

        # Create a dataframe for visualization
        df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
        df['label'] = labels
        df['Words'] = words_per_topic
        df['Size'] = frequencies

        # Create a scatter plot
        fig = px.scatter(df, x='x', y='y', color='label', size='Size', hover_data=['Words'],
                         labels={'color': 'Cluster', 'Words': 'Top words', 'Size': 'Cluster size'},
                         title='Cluster visualization', template='simple_white')

        fig.show()

from transformers import AutoAdapterModel
model_path = "distilbert-base-uncased"
adapter_path = "Elise-hf/distilbert-base-uncased_reddit_categories_unipelft"

model = AutoAdapterModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
adapter_name = model.load_adapter(adapter_path, source="hf", set_active=True)

# For Clustering we need to set the model to eval mode (Inference only)
model.eval()
dataset = datasets.load_dataset("Elise-hf/reddit_categories_clean")
full_dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
category = full_dataset['category']

le = LabelEncoder()
le.fit(category)

def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    # Concatenate title and selftext with [SEP] token in between
    combined_text = [title + ' [SEP] ' + selftext for title, selftext in zip(batch['title'], batch['text'])]

    # Encode the text using the tokenizer and add the attention mask
    encoding = tokenizer(combined_text, max_length=512, truncation=True, padding="longest")

    # Encode the labels and add them to the encoding
    labels = le.transform(batch['category'])
    encoding['labels'] = labels

    return encoding


dataset['validation'] = dataset['validation'].map(encode_batch, batched=True)
dataset['validation'].set_format(type="torch", columns=["input_ids", "attention_mask"])

clustering = Clustering(model,tokenizer)
embeddings = clustering.get_features(dataset['validation'],batch_size=32)
