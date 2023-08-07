import datasets
import html
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import textacy.preprocessing as tprep
import  argparse

RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')
REDDIT_DATASET_PATH = 'Elise-hf/reddit-self-post'


def get_raw_dataframe(dataset_path=None):
    """
    Convert a HuggingFace dataset into a pandas dataframe.
    """
    if dataset_path is None:
        dataset_path = REDDIT_DATASET_PATH

    ds = datasets.load_dataset(dataset_path, data_files='rspct.tsv')
    sub = datasets.load_dataset(dataset_path, data_files='subreddit_info.csv')
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
        'category_3': None,  # no data
        'in_data': None,  # not needed
        'reason_for_exclusion': None  # not needed
    }

    # define remaining columns
    columns = [c for c in column_mapping.keys() if column_mapping[c] != None]

    # select and rename those columns
    df = df[columns].rename(columns=column_mapping)
    return df


def clean(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    re.sub(r'\bu/\w+\b', '_USER_', text)
    re.sub(r'\br/\w+\b', '_SUBREDDIT_', text)

    return text.strip()


def textacy_clean(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    text = tprep.replace.urls(text)
    return text


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset_path', type=str, default=None)
    args.add_argument('--output_dir', type=str, defeault='output')
    args.add_argument('--output_name', type=str, defeault='reddit_data_cleaned')

    args = argparse.parse_args()
    df = get_raw_dataframe(REDDIT_DATASET_PATH)
    df['clean_text'] = df['text'].progress_apply(clean)
    df['clean_text'] = df['clean_text'].progress_apply(textacy_clean)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })

    print(f'Training data shape: {train_df.shape}')
    print(f'Testing data shape: {test_df.shape}')
    print(f'Val data shape: {val_df.shape}')

    # Save the data
    dataset.save_to_disk(args.output_dir)


if __name__ == '__main__':
    main()
