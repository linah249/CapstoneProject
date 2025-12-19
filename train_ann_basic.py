# train_ann_basic.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical, plot_model

from data_loader import load_basic_subject
from features import extract_features_from_raw

SAVE_DIR = "results_basic_ann"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 🔹 Load ALL basic subjects
# ------------------------------------------------------------
def load_all_basic():
    X_all, y_all = [], []

    for sid in range(1, 12):
        print(f"[BASIC] Loading subject {sid} ...")

        X_raw, y, _, _ = load_basic_subject(sid)
        feats = extract_features_from_raw(X_raw)

        X_all.append(feats)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print("\n[INFO] Total Basic windows loaded:", X_all.shape[0])
    print("[INFO] Unique Basic labels BEFORE cleaning:", np.unique(y_all))
    print()

    return X_all, y_all


# ------------------------------------------------------------
# 🔹 Build baseline ANN
# ------------------------------------------------------------
def build_ann(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ------------------------------------------------------------
# 🔹 Plot Helpers
# ------------------------------------------------------------
def plot_training_curves(history, name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("ANN Basic Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_acc.png"))
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("ANN Basic Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_loss.png"))
    plt.close()


def plot_conf_matrix(y_true, y_pred, classes, name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)

    plt.title("ANN Basic Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_cm.png"))
    plt.close()


# ------------------------------------------------------------
# 🔹 Train ANN Baseline
# ------------------------------------------------------------
def train_ann_basic():
    X, y = load_all_basic()

    # ------------------------------------------------------------
    # 🔥 FIX: Remove NaN labels
    # ------------------------------------------------------------
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    print("[INFO] Unique labels AFTER cleaning:", np.unique(y))

    # Classes
    classes = np.unique(y)
    num_classes = len(classes)

    # ------------------------------------------------------------
    # Normalize labels → 0, 1, ...
    # ------------------------------------------------------------
    label_map = {label: idx for idx, label in enumerate(classes)}
    y_mapped = np.array([label_map[v] for v in y])

    print("[INFO] Classes detected:", classes)
    print("[INFO] Label map:", label_map)

    # One-hot
    y_onehot = to_categorical(y_mapped, num_classes)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    model = build_ann(X.shape[1], num_classes)
    print(model.summary())

    # Try saving architecture
    try:
        plot_model(model, to_file=os.path.join(SAVE_DIR, "basic_ann_architecture.png"), show_shapes=True)
    except:
        pass

    # Train
    history = model.fit(
        X_tr, y_tr,
        epochs=25,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    plot_training_curves(history, "basic_ann")

    # Predict
    y_pred = np.argmax(model.predict(X_te), axis=1)
    y_true = np.argmax(y_te, axis=1)

    # Confusion matrix
    plot_conf_matrix(y_true, y_pred, classes, "basic_ann")

    # Save report
    report = classification_report(y_true, y_pred, target_names=[str(c) for c in classes])
    with open(os.path.join(SAVE_DIR, "basic_ann_report.txt"), "w") as f:
        f.write(report)

    print("\n===== CLASSIFICATION REPORT =====")
    print(report)

    model.save(os.path.join(SAVE_DIR, "basic_ann_model.h5"))
    print("\n[INFO] Saved ANN model.")


# ------------------------------------------------------------
# 🔹 RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== Training BASIC ANN Baseline ===")
    train_ann_basic()
