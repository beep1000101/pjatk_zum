"""
Model loading and preprocessing utilities for ASR commands evaluation.
"""
from pathlib import Path
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from datasets import Dataset, Audio
import json


def load_model_and_extractor(model_dir):
    model = AutoModelForAudioClassification.from_pretrained(model_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, feature_extractor, device


def load_test_dataset(splits_path):
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    labels = splits['labels']
    test_records = splits['splits']['test']
    test_df = [
        {'audio': r['path'], 'label': labels.index(r['label'])} for r in test_records
    ]
    test_ds = Dataset.from_list(test_df).cast_column(
        'audio', Audio(sampling_rate=16000))
    return test_ds, labels


def preprocess_test_dataset(test_ds, feature_extractor):
    def preprocess(batch):
        audio = batch['audio']
        out = feature_extractor(
            audio['array'], sampling_rate=audio['sampling_rate'])
        batch['input_values'] = out['input_values'][0]
        return batch
    return test_ds.map(preprocess)
