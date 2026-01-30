import numpy as np
import torch
from datasets import Audio, Dataset
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
from pathlib import Path


def run_inference(model, feature_extractor, device, test_ds):
    all_preds, all_labels = [], []
    n = len(test_ds)
    print(f"Running inference on {n} samples...")
    for i, ex in enumerate(test_ds):
        if i % 100 == 0 or i == n - 1:
            print(f"Sample {i+1}/{n}", flush=True)
        inp = torch.tensor(ex['input_values']).unsqueeze(0).to(device)
        logits = model(inp).logits.cpu().numpy()[0]
        pred = np.argmax(logits)
        all_preds.append(pred)
        all_labels.append(ex['label'])
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    report = classification_report(y_true, y_pred, labels=range(
        len(labels)), target_names=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


def load_test_dataset():
    outputs_dir = Path(__file__).parents[2] / \
        'outputs' / 'asr_commands' / 'preprocessing'
    splits_path = outputs_dir / 'splits.json'
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
