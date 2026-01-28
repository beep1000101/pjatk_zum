# Colab notebook (sentiment_embeddings)

This folder contains a **Colab-first** notebook used to run the full pipeline end-to-end with minimal local setup.

Why Colab?
- GPU/CPU resources are available by default.
- Installing `torch` / `transformers` / `datasets` is usually simpler than managing local wheels.

Notebook
- `colab_training.ipynb`

What it does
- Uses the project cache convention via `utils.paths.CACHE_PATH` (repo-root `/.cache`).
- Trains a baseline sentiment classifier on IMDB (`aclImdb`) using Hugging Face `Trainer` (DistilBERT).
- Writes model artifacts and metrics under `outputs/` and optionally zips the best checkpoint for download.

How to run in Colab (recommended)
1. Open Google Colab.
2. Clone the repo in a cell, then `cd` into it:
   - `!git clone <YOUR_REPO_URL>`
   - `%cd <REPO_DIR>`
3. Ensure the repo root is importable (so `from utils.paths import CACHE_PATH` works). If needed:
   - `import sys; sys.path.insert(0, '/content/<REPO_DIR>')`
4. Run the notebook cells top-to-bottom.

Notes
- Colab storage is ephemeral. If you want to persist `.cache/` and `outputs/`, mount Google Drive and copy or symlink those directories.
