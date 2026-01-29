"""
Model loading and preprocessing utilities for CLIP multimodal evaluation.
"""
from pathlib import Path
import torch
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
import json


def load_model_and_processor(model_name):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, processor, device


def load_test_dataset(splits_path):
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    labels = splits['labels']
    test_records = splits['splits']['test']
    test_df = pd.DataFrame([{'image': r['image'], 'text': r['text'],
                           'label': labels.index(r['label'])} for r in test_records])
    return test_df, labels
