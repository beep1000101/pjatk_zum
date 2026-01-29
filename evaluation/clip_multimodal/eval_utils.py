"""
Helper functions for CLIP multimodal evaluation.
"""
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


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
