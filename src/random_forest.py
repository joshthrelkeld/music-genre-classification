import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_f1(y_test, y_pred):
    # Extract per-genre F1 scores in a compact format for class-level comparison.
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = pd.DataFrame(report).T[["f1-score"]].round(2)
    f1 = f1.drop(index=["accuracy", "macro avg", "weighted avg"])
    return f1["f1-score"]

def run_random_forest():
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

    # Random forest is used to test whether genre boundaries are better captured
    # through nonlinear decision rules and interactions among audio features.
    # Feature scaling is not required, as the model is tree-based.
    rf_model = RandomForestClassifier(
        n_estimators=200,  # more trees reduce variance and stabilize predictions
        max_depth=None,    # allow full tree growth to capture complex structure
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # Accuracy summarizes overall performance, while the classification report
    # and confusion matrix show which genres are reliably separated and which
    # remain prone to overlap.
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    print("Random Forest F1 Scores:")
    print(get_f1(y_test, y_pred).to_string())

    return {
        "model_name": "Random Forest",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_scores": get_f1(y_test, y_pred)
    }

if __name__ == "__main__":
    run_random_forest()