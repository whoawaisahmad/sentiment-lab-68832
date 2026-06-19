"""Evaluation module: metrics, confusion matrices, timing."""

import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def evaluate_model(model, X_test, y_test, model_name, class_names, save_path=None):
    """
    Evaluate a trained model and plot the confusion matrix.

    Args:
        model: Trained sklearn estimator.
        X_test: Test feature matrix.
        y_test: True labels.
        model_name: String label for this model.
        class_names: List of class name strings.
        save_path: Optional path to save the confusion matrix image.

    Returns:
        Dict of metric values.
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0

    metrics = {
        'accuracy':         accuracy_score(y_test, y_pred),
        'precision_macro':  precision_score(y_test, y_pred, average='macro'),
        'recall_macro':     recall_score(y_test, y_pred, average='macro'),
        'f1_macro':         f1_score(y_test, y_pred, average='macro'),
        'f1_weighted':      f1_score(y_test, y_pred, average='weighted'),
        'inference_time_s': inference_time,
    }

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return metrics
