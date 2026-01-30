# Sentiment Embeddings Training & Evaluation

This notebook orchestrates both training and evaluation for the sentiment embeddings pipeline.

## Workflow
1. **Train**: Fine-tune a sequence classification model on the IMDB dataset.
2. **Evaluate**: After training, the notebook automatically runs evaluation on the test set using the exported model.

All helper functions and classes are defined in `helpers.py` for clarity and reproducibility.

## How to Use
- Run all cells in the notebook for end-to-end training and evaluation.
- Adjust hyperparameters or paths in the notebook as needed.

## Structure
- `helpers.py`: All utility functions for training and evaluation.
- `colab_training.ipynb`: Main notebook (orchestrates workflow).
