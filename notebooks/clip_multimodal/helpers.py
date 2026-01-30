import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path


def run_inference(model, processor, device, images, text_features, batch_size=64, log_every=5):
    n = len(images)
    preds = []
    import time
    t0 = time.perf_counter()
    n_batches = (n + batch_size - 1) // batch_size
    for batch_i, start in enumerate(range(0, n, batch_size), start=1):
        end = min(start + batch_size, n)
        batch_imgs = [Image.fromarray(images[i]) for i in range(start, end)]
        img_inputs = processor(images=batch_imgs, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(device)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=pixel_values)
            img_features = model.visual_projection(vision_out.pooler_output)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        logits = img_features @ text_features.T
        preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        if batch_i % log_every == 0 or end == n:
            now = time.perf_counter()
            elapsed = now - t0
            done = end
            rate = done / elapsed if elapsed > 0 else float("inf")
            remaining = (n - done) / rate if rate > 0 else float("inf")
            print(
                f"[{batch_i}/{n_batches}] {done}/{n} images | {rate:.1f} img/s | ETA {remaining/60:.1f} min")
    return np.array(preds)


def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    return {"top1_accuracy": float(acc)}


def load_cifar10_test():
    from utils.paths import CACHE_PATH
    raw_dir = CACHE_PATH / "clip_multimodal" / "raw" / "cifar-10-batches-py"
    test_batch_path = raw_dir / "test_batch"
    with open(test_batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_true = np.array(batch[b"labels"], dtype=np.int64)
    return images, y_true
