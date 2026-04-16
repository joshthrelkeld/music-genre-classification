# Central training script for the music genre classification project.
# Runs all core models and prints a summary comparison.

import pandas as pd
from logistic_regression import run_logistic_regression
from random_forest import run_random_forest
from svm import run_svm

def main():
    print("\nStarting music genre classification workflow...\n")

    print("=" * 60)
    print("Running Logistic Regression")
    print("=" * 60)
    lr_results = run_logistic_regression()

    print("\n" + "=" * 60)
    print("Running Random Forest")
    print("=" * 60)
    rf_results = run_random_forest()

    print("\n" + "=" * 60)
    print("Running SVM")
    print("=" * 60)
    svm_results = run_svm()

    print("\n" + "=" * 60)
    print("Model Accuracy Summary")
    print("=" * 60)
    print(f"{lr_results['model_name']}: {lr_results['accuracy']:.4f}")
    print(f"{rf_results['model_name']}: {rf_results['accuracy']:.4f}")
    print(f"{svm_results['model_name']}: {svm_results['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("Per-Genre F1 Score Comparison")
    print("=" * 60)
    f1_comparison = pd.DataFrame({
        lr_results['model_name']: lr_results['f1_scores'],
        rf_results['model_name']: rf_results['f1_scores'],
        svm_results['model_name']: svm_results['f1_scores'],
    })
    print(f1_comparison.to_string())

if __name__ == "__main__":
    main()