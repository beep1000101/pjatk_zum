from pathlib import Path
import pandas as pd
import json
from datasets import Dataset, Audio
from utils.paths import CACHE_PATH


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
