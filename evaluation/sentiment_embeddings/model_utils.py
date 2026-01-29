"""
Model loading and preprocessing utilities for sentiment embeddings evaluation.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import json


def load_model_and_tokenizer(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, tokenizer, device


def load_test_dataset(splits_path):
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    labels = splits['labels']
    test_records = splits['splits']['test']
    test_df = pd.DataFrame(
        [{'text': r['text'], 'label': labels.index(r['label'])} for r in test_records])
    return test_df, labels
