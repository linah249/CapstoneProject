# train_baselines.py
"""
Train classical baselines (kNN, SVM, RandomForest) on extracted features
using subject-transfer split.
"""

import os
import joblib
import numpy as np
import pandas as pd
import sys # <--- ADDED: Fixes NameError: name 'sys' is not defined
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- FIXED IMPORTS ---
# 1. Import the entire features module to access its functions via 'features.function_name'
import features 
# 2. We assume this import is correct based on your project structure
from utils.plots import plot_confusion_matrix 

    
# --- CONFIGURATION (Self-contained) ---
TRAIN_SUBJECTS = list(range(1, 9))   
TEST_SUBJECTS = [9, 10, 11]         
BASELINE_RESULTS = "../results/baseline_results" 
LABEL_MAP = { 
    1: "Flexion", 2: "Extension", 3: "Supination", 4: "Pronation",
    5: "Open Hand", 6: "Pinch", 7: "Lateral Pinch", 8: "Fist"
}
LABELS_ORDER = sorted(LABEL_MAP.keys())

# --- Create directory for results ---
os.makedirs(BASELINE_RESULTS, exist_ok=True)

MODELS = {
    "RandomForest": Pipeline([("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
    "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=42))]),
    "kNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
}

def save_classification_report(report_dict, out_path):
    """Save classification report dict as CSV."""
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(out_path, index=True)

if __name__ == "__main__":
    print("[1] Load features ...")
    
    # --- FIX: Call the feature loader via the 'features' module ---
    try:
        X_all, df_all = features.load_all_subjects_features()
    except AttributeError as e:
        print(f"\nFATAL ERROR: Failed to find feature loading function: {e}")
        print("Please ensure 'load_all_subjects_features' is correctly defined (name and capitalization) in your features.py file.")
        # We replace the missing 'sys.exit(1)' with a normal exit after logging the error
        sys.exit(1)
    
    labels = df_all['label'].values.astype(int)
    subjects = df_all['subject'].values.astype(int)

    train_mask = np.isin(subjects, TRAIN_SUBJECTS)
    test_mask = np.isin(subjects, TEST_SUBJECTS)

    X_train, y_train = X_all[train_mask], labels[train_mask]
    X_test, y_test = X_all[test_mask], labels[test_mask]

    print(f"[1] Data loaded. Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"[1] Feature dimension: {X_all.shape[1]}") 
    
    results = []

    for name, model in MODELS.items():
        print(f"\n[TRAIN] {name}")

        model_dir = os.path.join(BASELINE_RESULTS, name)
        os.makedirs(model_dir, exist_ok=True)

        # Train
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))

        # Predict
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Save accuracy in summary list
        results.append({"model": name, "accuracy": acc})

        # Classification Report
        cr = classification_report(y_test, y_pred, target_names=list(LABEL_MAP.values()), output_dict=True)
        save_classification_report(cr, os.path.join(model_dir, f"{name}_classification_report.csv"))

        # Also save readable text version
        with open(os.path.join(model_dir, f"{name}_classification_report.txt"), "w") as f:
            f.write(classification_report(y_test, y_pred, target_names=list(LABEL_MAP.values())))

        # Confusion matrix (matrix values)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS_ORDER)
        pd.DataFrame(cm, index=LABEL_MAP.values(), columns=LABEL_MAP.values()).to_csv(
            os.path.join(model_dir, f"{name}_confusion_matrix.csv")
        )

        # Confusion matrix plot
        plot_confusion_matrix(
            y_test,
            y_pred,
            labels=list(LABEL_MAP.values()),
            out_path=os.path.join(model_dir, f"{name}_cm.png"),
            normalize=True,
            title=f"{name} Confusion Matrix"
        )

        print(f"[saved] Results stored in: {model_dir}")

    # Save summary across all models
    pd.DataFrame(results).to_csv(os.path.join(BASELINE_RESULTS, "baseline_summary.csv"), index=False)
    print("[Done] Baseline results saved to", BASELINE_RESULTS)