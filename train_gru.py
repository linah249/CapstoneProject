"""
train_gru_stream_allsubjects.py (FINAL CORRECTED VERSION)
Memory-safe GRU training across all subjects (streams CSV rows) 
with crucial RAW DATA SCALING applied. Plots are correctly defined.
"""

import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

# -----------------------------
# USER CONFIG — update if needed
# -----------------------------
DATA_DIR = r"C:\Users\linah\Desktop\Final Year Project Linah\sEMG dataset\Basic Movements"
SAVE_DIR = r"C:\Users\linah\Desktop\Final Year Project Linah\results\gru_allsubjects"
MODELS_DIR = r"C:\Users\linah\Desktop\Final Year Project Linah\results\ann_advanced\models" # Where scaler is saved
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "figs"), exist_ok=True)

TRAIN_SUBJECTS = list(range(1,9))   
TEST_SUBJECTS  = [9, 10, 11]

BATCH_SIZE = 32
EPOCHS = 15
STEPS_PER_EPOCH = 200
VAL_STEPS = 50

LABEL_MAP = {
    1:"Flexion",2:"Extension",3:"Supination",4:"Pronation",
    5:"Open Hand",6:"Pinch",7:"Lateral Pinch",8:"Fist"
}
NUM_CLASSES = 8

# Load the pre-fitted scaler for raw EMG
RAW_SCALER = None
try:
    RAW_SCALER = joblib.load(os.path.join(MODELS_DIR, "scaler_raw_emg.joblib")) 
    print("[INFO] Successfully loaded RAW_SCALER.")
except FileNotFoundError:
    print("[WARNING] RAW_SCALER not found. Training on unscaled data (risky).")


# -----------------------------
# Utility: list CSVs and map subject id (UNCHANGED)
# -----------------------------
def list_subject_files(data_dir):
    pattern = os.path.join(data_dir, "Nearlab_sbj*_WL512_S128*.csv")
    files = sorted(glob.glob(pattern))
    subj_map = {}
    import re
    for f in files:
        basename = os.path.basename(f)
        m = re.search(r"sbj(\d+)", basename, re.IGNORECASE)
        if m:
            sid = int(m.group(1))
            subj_map[sid] = f
    return subj_map

# -----------------------------
# Stream single CSV rows (generator) (UNCHANGED)
# -----------------------------
def stream_rows_from_file(path):
    while True:
        with open(path, "r") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 5121:
                    continue
                # Note: DeprecationWarning for np.fromstring is harmless here
                raw = np.fromstring(",".join(parts[:5120]), dtype=np.float32, sep=",")
                if raw.size < 5120:
                    continue
                raw = raw.reshape(10, 512)  # (channels, samples)
                label = int(float(parts[5120]))
                yield raw, label

# -----------------------------
# Streaming Sequence for Keras (UPDATED with SCALER)
# -----------------------------
class StreamSequence(Sequence):
    def __init__(self, file_list, batch_size=32):
        self.files = list(file_list)
        if not self.files:
            raise ValueError("No files provided")
        self.iters = [stream_rows_from_file(p) for p in self.files]
        self.batch_size = batch_size

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        Xb = []
        yb = []
        while len(Xb) < self.batch_size:
            i = random.randrange(len(self.iters))
            try:
                raw, label = next(self.iters[i])
            except StopIteration:
                self.iters[i] = stream_rows_from_file(self.files[i])
                raw, label = next(self.iters[i])

            # Apply Scaler here: (10, 512) -> (5120,) -> Scale -> (10, 512)
            if RAW_SCALER is not None:
                raw_flat = raw.reshape(1, -1)
                scaled_flat = RAW_SCALER.transform(raw_flat) 
                raw = scaled_flat.reshape(10, 512) 
            
            # convert to RNN input shape (timesteps, features) = (512, 10)
            sample = np.transpose(raw, (1,0)).astype(np.float32)
            Xb.append(sample)
            yb.append(label-1) 
            
        Xb = np.stack(Xb, axis=0) 
        yb = to_categorical(yb, num_classes=NUM_CLASSES)
        return Xb, yb

# -----------------------------
# Build GRU (UNCHANGED)
# -----------------------------
def build_gru_model():
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(512,10)),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------
# Plot helpers (FIXED: ensures all functions are defined)
# -----------------------------
def save_fig(fig, name):
    p = os.path.join(SAVE_DIR, "figs", name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("[saved]", p)

def plot_history(history, name):
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].plot(history.history["loss"], label="train")
    axs[0].plot(history.history.get("val_loss", []), label="val")
    axs[0].set_title("Loss"); axs[0].legend()
    axs[1].plot(history.history["accuracy"], label="train")
    axs[1].plot(history.history.get("val_accuracy", []), label="val")
    axs[1].set_title("Accuracy"); axs[1].legend()
    save_fig(fig, f"{name}_training.png")
    plt.close(fig)

def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1,NUM_CLASSES+1)))
    cmn = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-12)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=list(LABEL_MAP.values()), yticklabels=list(LABEL_MAP.values()), ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    save_fig(fig, f"{name}_cm.png")
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    subj_map = list_subject_files(DATA_DIR)
    if not subj_map:
        raise RuntimeError("No subject CSV files found in DATA_DIR. Check pattern.")

    train_files = [subj_map[sid] for sid in subj_map.keys() if sid in TRAIN_SUBJECTS and sid in subj_map]
    test_files  = [subj_map[sid] for sid in subj_map.keys() if sid in TEST_SUBJECTS and sid in subj_map]

    print("Train files:", train_files)
    print("Test files:", test_files)

    train_seq = StreamSequence(train_files, batch_size=BATCH_SIZE)
    val_seq   = StreamSequence(test_files,  batch_size=BATCH_SIZE)

    model = build_gru_model()
    model.summary()

    best_path = os.path.join(SAVE_DIR, "gru_best.keras")
    callbacks = [
        # Note: Your model is showing high training accuracy but low validation accuracy (overfitting/generalization issue).
        # EarlyStopping and ReduceLROnPlateau will help manage this.
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        train_seq,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_seq,
        validation_steps=VAL_STEPS,
        callbacks=callbacks,
        verbose=2
    )

    model.save(os.path.join(SAVE_DIR, "gru_final.keras"))
    plot_history(history, "gru") # <-- This call is now safe

    try:
        plot_model(model, to_file=os.path.join(SAVE_DIR, "figs", "gru_architecture.png"), show_shapes=True)
    except Exception as e:
        print("plot_model failed:", e)

    # sample evaluation
    y_true, y_pred = [], []
    for _ in range(VAL_STEPS):
        Xb, yb = val_seq.__getitem__(0)
        probs = model.predict(Xb, verbose=0)
        preds = (np.argmax(probs, axis=1) + 1).tolist()
        truths = (np.argmax(yb, axis=1) + 1).tolist()
        y_pred.extend(preds); y_true.extend(truths)

    print(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values())))
    plot_confusion(y_true, y_pred, "gru_sampled_eval")

    acc = accuracy_score(y_true, y_pred)
    pd = __import__("pandas")
    pd.DataFrame([{"model":"GRU","accuracy":float(acc)}]).to_csv(os.path.join(SAVE_DIR,"summary_gru.csv"), index=False)
    print("[DONE] GRU training + sampled evaluation complete. Results in", SAVE_DIR)

if __name__ == "__main__":
    main()