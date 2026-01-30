import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def run_inference(model, tokenizer, device, test_df):
    all_preds, all_labels = [], []
    n = len(test_df)
    print(f"Running inference on {n} samples...")
    with torch.no_grad():
        for i, (_, row) in enumerate(test_df.iterrows()):
            if i % 100 == 0 or i == n - 1:
                print(f"Sample {i+1}/{n}", flush=True)
            inputs = tokenizer(
                row['text'], return_tensors='pt', truncation=True, padding=True).to(device)
            logits = model(**inputs).logits.cpu().numpy()[0]
            pred = np.argmax(logits)
            all_preds.append(pred)
            all_labels.append(row['sentiment_value'])
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


def find_hf_model_dir(root_dir):
    import os
    import json
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'config.json' in filenames:
            config_path = os.path.join(dirpath, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if 'model_type' in config:
                return dirpath
    raise FileNotFoundError('No valid Hugging Face model directory found.')
