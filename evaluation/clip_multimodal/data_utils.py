import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from utils.paths import CACHE_PATH


def load_cifar10_test():
    raw_dir = CACHE_PATH / "clip_multimodal" / "raw" / "cifar-10-batches-py"
    test_batch_path = raw_dir / "test_batch"
    with open(test_batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_true = np.array(batch[b"labels"], dtype=np.int64)
    return images, y_true
