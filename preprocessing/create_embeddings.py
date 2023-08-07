from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoAdapterModel
from transformers import AdapterType
from sklearn.preprocessing import LabelEncoder
import datasets
import torch
import os
import numpy as np
import pandas as pd
import argparse
import logging
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batch_embeddings(batch):
    """
    Computes embeddings for a batch of input data.

    Args:
        batch (dict): A dictionary containing the input data. Must have keys 'input_ids' and 'attention_mask'.

    Returns:
        dict: A dictionary containing the computed embeddings.
    """
    with torch.no_grad():
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        embeddings = encoder.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        return {"embeddings_pretrained": embeddings.detach().cpu().numpy()}


def tokenizde_dataset(dataset,encoder, tokenizer):
    """
    Tokenizes a dataset and encodes the labels.

    Args:
        dataset (datasets.Dataset): The dataset to tokenize and encode.
        encoder (AutoAdapterModel): The adapter model to use for encoding.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.

    Returns:
        datasets.Dataset: The tokenized and encoded dataset.
    """

    def encode_batch(batch):

            # Concatenate title and selftext with [SEP] token in between
            combined_text = [title + ' [SEP] ' + selftext for title, selftext in zip(batch['title'], batch['text'])]

            # Encode the text using the tokenizer and add the attention mask
            encoding = tokenizer(combined_text, max_length=512, truncation=True, padding="longest")


            # Encode the labels and add them to the encoding
            encoding['subcategory'] = subcategory_encoder.transform(batch['subcategory'])
            encoding['category'] = category_encoder.transform(batch['category'])
            return encoding
    
    dataset.set_format( type="torch", columns=["title", "text", "subcategory", "category"] ,device=device)
    
    dataset = dataset.map( encode_batch, batched=True, desc=' Encoding')
    dataset.reset_format()
    return dataset


def main():

    parser = argparse.ArgumentParser(description='Create embeddings for Reddit categories')
    parser.add_argument('--adapter_path', type=str, default=ADAPTER_PATH, help='Path to adapter weights')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for computing embeddings')
    parser.add_argument('--num_proc', type=int, default=6, help='Number of processes to use for saving embeddings')
    parser.add_argument('--output_dir', type=str, default='data/reddit_categories_clean_embeddings_with_pretrained', help='Path to save embeddings')
    parser.add_argument('--set_active', type=bool, default=True, help='Set the adapter as active')
    parser.add_argument('--split', default=None, nargs='+', help='Splits to use for creating embeddings. If None, all splits are used for example: --split train test')
    args = parser.parse_args()

    # Load the model and tokenizer and the adapter
    encoder = AutoAdapterModel.from_pretrained("distilbert-base-uncased",cache_dir='model_weights')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    adapter_name = encoder.load_adapter(args.adapter_path, source="hf",set_active=args.set_active)

    # Load the dataset
    if args.split is None:
        dataset = datasets.load_dataset("data/reddit_categories_clean",name="reddit_categories_clean_embeddings")
    else:
        dataset = datasets.load_dataset("data/reddit_categories_clean",name="reddit_categories_clean_embeddings", split=args.split)

    # Encode the labels
    tokenizde_dataset = tokenizde_dataset(dataset,encoder, tokenizer)

    # Compute the embeddings
    dataset.set_format( type="torch", columns=["input_ids", "attention_mask"] ,device=device)

    dataset = dataset.map(batch_embeddings, batched=True,batch_size=args.batch_size, desc=' Computing embeddings')
    dataset.reset_format()
    dataset.save_to_disk(args.output_dir,num_proc=args.num_proc)

if __name__ == '__main__':
    main()
