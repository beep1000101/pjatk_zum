# CLIP Multimodal Training & Evaluation

This notebook orchestrates both training and evaluation for the CLIP multimodal pipeline.

## Workflow
1. **Train**: Use CLIP to perform zero-shot or fine-tuned classification on CIFAR-10.
2. **Evaluate**: After training, the notebook automatically runs evaluation on the test set using the model and processor.

All helper functions and classes are defined in `helpers.py` for clarity and reproducibility.

## How to Use
- Run all cells in the notebook for end-to-end training and evaluation.
- Adjust hyperparameters or paths in the notebook as needed.

## Structure
- `helpers.py`: All utility functions for training and evaluation.
- `colab_training.ipynb`: Main notebook (orchestrates workflow).
