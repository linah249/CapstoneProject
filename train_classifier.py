# train_classifier.py
"""
Train & evaluate multiple classifiers using subject-transfer split.
Usage:
    - Place this file in your code/ folder
    - Run in Spyder (working dir = code/)
    - Requires features.py and data_loader.py (already created)
Outputs:
    - Saved models in ./results/models/
    - Plots saved to ./results/figs/
    - Summary CSV ./results/summary_results.csv
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import feature-loading function from features.py
from features import load_all_subjects_features

# -----------------------
# Config
# -----------------------
RESULTS_DIR = "../results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FIGS_DIR = os.path.join(RESULTS_DIR, "figs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# Subject split (subject-transfer)
TRAIN_SUBJECTS = list(range(1, 9))   # 1..8
TEST_SUBJECTS = [9, 10, 11]         # 9..11

RANDOM_STATE = 42

# Models to train (Pipeline with scaler where appropriate)
MODELS = {
    "RandomForest": Pipeline([("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))]),
    "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE))]),
    "kNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]),
    "MLP": Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, random_state=RANDOM_STATE))]),
}

# Label mapping (for plots)
LABEL_MAP = {
    1: "Flexion",
    2: "Extension",
    3: "Supination",
    4: "Pronation",
    5: "Open Hand",
    6: "Pinch",
    7: "Lateral Pinch",
    8: "Fist"
}
LABELS_ORDER = sorted(LABEL_MAP.keys())

# -----------------------
# Helpers: plotting and evaluation
# -----------------------
def save_fig(fig, fname):
    out = os.path.join(FIGS_DIR, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[save_fig] {out}")

def plot_model_comparison(summary_df):
    # bar chart for accuracy and f1_macro
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(summary_df))
    ax.bar(x - 0.15, summary_df['accuracy'], width=0.3, label='Accuracy')
    ax.bar(x + 0.15, summary_df['f1_macro'], width=0.3, label='Macro F1')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['model'], rotation=20)
    ax.set_ylim(0,1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (subject-transfer: train 1-8, test 9-11)")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "model_comparison_accuracy_f1.png")
    plt.show()

def plot_confusion_matrix(cm, labels, title="Confusion matrix", outname=None, normalize=True):
    if normalize:
        cmn = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    else:
        cmn = cm
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cmn, annot=True, fmt='.2f' if normalize else 'd', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if outname:
        save_fig(fig, outname)
    plt.show()

def per_class_accuracy_from_confusion(cm):
    # cm: confusion matrix (rows true, cols pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-12)
    return acc_per_class

# -----------------------
# Main pipeline
# -----------------------
def run_subject_transfer_experiment(train_subjects=TRAIN_SUBJECTS, test_subjects=TEST_SUBJECTS):
    # 1) Load features (all subjects)
    print("[1] Loading features for all subjects (this may take a bit)...")
    X_all, df_all = load_all_subjects_features()  # verbose inside
    # df_all contains 'label' column and 'subject'
    labels_all = df_all['label'].values.astype(int)
    subjects_all = df_all['subject'].values.astype(int)

    # 2) Split by subject ids
    train_mask = np.isin(subjects_all, train_subjects)
    test_mask = np.isin(subjects_all, test_subjects)

    X_train = X_all[train_mask]
    y_train = labels_all[train_mask]
    X_test = X_all[test_mask]
    y_test = labels_all[test_mask]

    print(f"[2] Split sizes -> Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"[2] Unique train subjects: {np.unique(subjects_all[train_mask])}")
    print(f"[2] Unique test subjects: {np.unique(subjects_all[test_mask])}")

    # 3) Train & evaluate models
    results = []
    models_trained = {}
    for name, pipe in MODELS.items():
        print(f"\n[3] Training model: {name}")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        # save model
        model_fname = os.path.join(MODELS_DIR, f"{name}_subject_transfer.joblib")
        joblib.dump(pipe, model_fname)

        # predict
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, digits=4, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS_ORDER)

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1m,
            "train_time_s": train_time,
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0]
        })
        models_trained[name] = {
            "pipeline": pipe,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_pred": y_pred
        }
        print(f"[3] {name} -> acc={acc:.4f}, f1_macro={f1m:.4f}, train_time={train_time:.1f}s")

    # 4) Summary dataframe and save
    df_results = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    df_results.to_csv(os.path.join(RESULTS_DIR, "summary_results.csv"), index=False)
    print(f"\n[4] Saved summary to {os.path.join(RESULTS_DIR, 'summary_results.csv')}")
    print(df_results)

    # 5) Plots: model comparison
    plot_model_comparison(df_results)

    # 6) Confusion matrices for each model (normalized)
    for name, info in models_trained.items():
        cm = info['confusion_matrix']
        plot_confusion_matrix(cm, [LABEL_MAP[l] for l in LABELS_ORDER],
                              title=f"{name} - Normalized Confusion Matrix",
                              outname=f"cm_{name}.png", normalize=True)

    # 7) Per-class accuracy for best model (highest accuracy)
    best_model_name = df_results.iloc[0]['model']
    best_info = models_trained[best_model_name]
    cm_best = best_info['confusion_matrix']
    acc_per_class = per_class_accuracy_from_confusion(cm_best)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar([LABEL_MAP[l] for l in LABELS_ORDER], acc_per_class)
    ax.set_ylim(0,1)
    ax.set_ylabel("Per-class accuracy")
    ax.set_title(f"Per-class accuracy (best model: {best_model_name})")
    plt.xticks(rotation=25)
    plt.tight_layout()
    save_fig(fig, f"per_class_accuracy_{best_model_name}.png")
    plt.show()

    # 8) Save detailed reports (classification_report) for each model
    for name, info in models_trained.items():
        cr = info['classification_report']
        cr_df = pd.DataFrame(cr).transpose()
        cr_df.to_csv(os.path.join(RESULTS_DIR, f"classification_report_{name}.csv"))
        print(f"[save] classification_report_{name}.csv")

    print("\n[Done] Subject-transfer experiment complete.")
    return df_results, models_trained

# -----------------------
# Run if called directly
# -----------------------
if __name__ == "__main__":
    df_results, models_trained = run_subject_transfer_experiment()

