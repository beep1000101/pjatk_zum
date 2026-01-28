# Colab notebook (asr_commands)

This folder contains a **Colab-first** notebook used to train and evaluate a small keyword-spotting model on TensorFlow mini speech commands.

Why Colab?
- Training and feature extraction are noticeably faster.
- Installing audio deps (PyTorch + torchaudio) is often easier than on some local setups.

Notebook
- `colab_training.ipynb`

What it does
- Ensures the dataset is available by running `python data_ingestion/asr_commands/run.py` if needed.
- Builds deterministic per-label splits (train/val/test) and saves:
  - `outputs/asr_commands/preprocessing/splits.json`
- Computes log-mel features on the fly with `torchaudio`.
- Trains a small CNN baseline and saves best checkpoint:
  - `outputs/asr_commands/best/kws_cnn.pt`
  - training metrics: `outputs/asr_commands/training/metrics.json`
- Evaluates on the test split and writes:
  - `outputs/asr_commands/evaluation/metrics.json` (includes confusion matrix)
- Optionally zips outputs for download.

How to run in Colab (recommended)
1. Open Google Colab.
2. Clone the repo and `cd` into it:
   - `!git clone <YOUR_REPO_URL>`
   - `%cd <REPO_DIR>`
3. If `torchaudio` is missing, install it in the Colab runtime and restart the kernel.
4. Run the notebook cells top-to-bottom.

Notes
- Colab storage is ephemeral. Mount Google Drive if you want to persist `.cache/` and `outputs/`.
