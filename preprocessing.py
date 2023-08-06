import datasets
import numpy as np
import pandas as pd

REDDIT_DATASET_PATH = 'Elise-hf/reddit-self-post'

def get_dataset(dataset_name):
    """
    Load a dataset from the HuggingFace Datasets library.
    """
    dataset = datasets.load_dataset(dataset_name)
    return dataset

def get_raw_dataframe(dataset_path=None):
    """
    Convert a HuggingFace dataset into a pandas dataframe.
    """
    if dataset_path is None:
        dataset_path = REDDIT_DATASET_PATH

    ds = datasets.load_dataset(dataset_path,data_files='rspct.tsv')
    sub = datasets.load_dataset(dataset_path,data_files='subreddit_info.csv')
    subred_df = sub['train'].to_pandas().set_index(['subreddit'])
    df = ds['train'].to_pandas()
    df = df.join(subred_df, on='subreddit')
    column_mapping = {
    'id': 'id',
    'subreddit': 'subreddit',
    'title': 'title',
    'selftext': 'text',
    'category_1': 'category',
    'category_2': 'subcategory',
    'category_3': None, # no data
    'in_data': None, # not needed
    'reason_for_exclusion': None # not needed
    }

    # define remaining columns
    columns = [c for c in column_mapping.keys() if column_mapping[c] != None]

    # select and rename those columns
    df = df[columns].rename(columns=column_mapping)
    return df