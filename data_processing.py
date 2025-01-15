import pandas as pd
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text


def load_and_prepare_data(fake_path, true_path):
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels
    fake_df['label'] = 0
    true_df['label'] = 1

    # Combine and shuffle
    data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
    data = shuffle(data).reset_index(drop=True)

    # Clean text
    data['text'] = data['text'].apply(clean_text)

    # Split into train and test
    return train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)


def tokenize_data(texts, max_length=64):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
