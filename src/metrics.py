"""Shared evaluation utilities for model comparison."""
import pandas as pd
from sklearn.metrics import classification_report


def get_f1(y_test, y_pred):
    """Extract per-genre F1 scores in a compact format for class-level comparison."""
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = pd.DataFrame(report).T[["f1-score"]].round(2)
    f1 = f1.drop(index=["accuracy", "macro avg", "weighted avg"])
    return f1["f1-score"]