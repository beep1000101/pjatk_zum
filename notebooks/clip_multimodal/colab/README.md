# Colab notebook (clip_multimodal)

This folder contains a **Colab-first** notebook used to evaluate a pretrained CLIP model on CIFAR-10.

Why Colab?
- Fast CPU/GPU runtime for embedding inference.
- Easy access to `torch` + `transformers`.

Notebook
- `colab_training.ipynb`

What it does
- Ensures CIFAR-10 is available by running `python data_ingestion/clip_multimodal/run.py` if needed.
- Loads CIFAR-10 test batch from `.cache/clip_multimodal/raw/cifar-10-batches-py/`.
- Downloads a pretrained CLIP checkpoint (default: `openai/clip-vit-base-patch32`).
- Computes prompt-based top-1 accuracy for prompts `"a photo of a {label}"`.
- Writes outputs to:
  - `outputs/clip_multimodal/evaluation/metrics.json`
  - `outputs/clip_multimodal/evaluation/sample_predictions.csv`
  - optional zip: `clip_multimodal_outputs.zip`

How to run in Colab (recommended)
1. Open Google Colab.
2. Clone the repo and `cd` into it:
   - `!git clone <YOUR_REPO_URL>`
   - `%cd <REPO_DIR>`
3. Run the notebook cells top-to-bottom.

Notes
- Colab storage is ephemeral. Consider mounting Google Drive if you want to keep the outputs.
