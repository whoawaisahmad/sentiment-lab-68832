"""Feature extraction module: BoW and TF-IDF vectorizers."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_bow_vectorizer(max_features=5000):
    """
    Return a configured Bag-of-Words CountVectorizer.

    Args:
        max_features: Maximum vocabulary size.

    Returns:
        Configured CountVectorizer instance.
    """
    return CountVectorizer(max_features=max_features)


def get_tfidf_vectorizer(max_features=5000):
    """
    Return a configured TF-IDF vectorizer with sublinear TF scaling.

    Args:
        max_features: Maximum vocabulary size.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(max_features=max_features, sublinear_tf=True)


def print_top_terms(vectorizer, X, y, label_map, n=20):
    """
    Print the top N terms per class by mean weight.

    Args:
        vectorizer: Fitted vectorizer instance.
        X: Feature matrix (sparse).
        y: Label array.
        label_map: Dict mapping class index to class name.
        n: Number of top terms to display.
    """
    feature_names = vectorizer.get_feature_names_out()
    y_arr = np.array(y)
    for cls_idx, cls_name in label_map.items():
        mask = (y_arr == cls_idx)
        mean_weights = X[mask].toarray().mean(axis=0)
        top_indices = np.argsort(mean_weights)[::-1][:n]
        print(f"\nTop {n} terms for class '{cls_name}':")
        print([feature_names[i] for i in top_indices])
