import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_f1(y_test, y_pred):
    """Extract per-genre F1 scores in a compact format for class-level comparison."""
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = pd.DataFrame(report).T[["f1-score"]].round(2)
    f1 = f1.drop(index=["accuracy", "macro avg", "weighted avg"])
    return f1["f1-score"]

def run_svm():
    # Use a file path based on the script location so the code runs reliably
    # across environments rather than depending on the working directory.
    data_path = Path(__file__).resolve().parent.parent / "data" / "data.csv"
    df = pd.read_csv(data_path)

    # Exclude the file identifier and target label so the model trains only on
    # the audio-derived predictors.
    X = df.drop(columns=["filename", "label"])
    y = df["label"]

    # Stratification preserves genre proportions across train and test splits,
    # making evaluation less sensitive to accidental class imbalance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # SVM is sensitive to feature scale, so the inputs are standardized first.
    # The RBF kernel is used to test whether genre classes are separated by
    # nonlinear decision boundaries that a linear model may fail to capture.
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",   # allows curved decision boundaries
            C=10,           # penalizes misclassification more strongly, improving fit at the cost of higher overfitting risk
            gamma="scale",  # adapts kernel sensitivity to the feature distribution
            random_state=42
        ))
    ])

    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # Accuracy summarizes overall performance, while the classification report
    # and confusion matrix show which genres are separable and where overlap
    # remains despite a nonlinear classifier.
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    print("SVM F1 Scores:")
    print(get_f1(y_test, y_pred).to_string())

    return {
        "model_name": "SVM",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_scores": get_f1(y_test, y_pred)
    }

if __name__ == "__main__":
    run_svm()