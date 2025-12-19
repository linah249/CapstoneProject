"""
train_ann_advanced.py

Advanced ANN + 1D-CNN pipeline for Nearlab sEMG (subject-transfer).
- Trains Dense ANN on extracted features (50-dim)
- Optionally trains 1D-CNN on raw windows (10x512)
- Uses data augmentation, class weighting, early stopping, LR schedule
- Produces many publication-style plots for your report
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, BatchNormalization, GlobalAveragePooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import regularizers

# Import your existing code (features & data loader)
from features import load_all_subjects_features, plot_pca_embedding
from data_loader import load_basic_subject, drop_nan_labels

# ---------------------------
# Config
# ---------------------------
RESULTS_BASE = "../results/ann_advanced"
os.makedirs(RESULTS_BASE, exist_ok=True)
MODELS_DIR = os.path.join(RESULTS_BASE, "models")
FIGS_DIR = os.path.join(RESULTS_BASE, "figs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

TRAIN_SUBJECTS = list(range(1, 9))  # 1..8
TEST_SUBJECTS = [9, 10, 11]

RND = 42
np.random.seed(RND)
tf.random.set_seed(RND)

LABEL_MAP = {
    1: "Flexion", 2: "Extension", 3: "Supination", 4: "Pronation",
    5: "Open", 6: "Pinch", 7: "LateralPinch", 8: "Fist"
}
LABEL_ORDER = sorted(LABEL_MAP.keys())

# ---------------------------
# Utility plotting functions
# ---------------------------
sns.set(style="whitegrid", context="notebook", rc={"figure.figsize": (8,6)})

def save_fig(fig, name):
    path = os.path.join(FIGS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[saved] {path}")

def plot_training_history(history, name_prefix="ann"):
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    ax[0].plot(history.history['loss'], label='train loss')
    ax[0].plot(history.history['val_loss'], label='val loss')
    ax[0].legend(); ax[0].set_title('Loss')
    ax[1].plot(history.history['accuracy'], label='train acc')
    ax[1].plot(history.history['val_accuracy'], label='val acc')
    ax[1].legend(); ax[1].set_title('Accuracy')
    save_fig(fig, f"{name_prefix}_training_curves.png")
    plt.show()

def plot_confusion(y_true, y_pred, labels=LABEL_ORDER, title="Confusion Matrix", outname="cm.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm.astype('float') / (cm.sum(axis=1)[:, None] + 1e-12)
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[LABEL_MAP[l] for l in labels],
                yticklabels=[LABEL_MAP[l] for l in labels], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    save_fig(fig, outname)
    plt.show()
    return cm

def per_class_accuracy(cm):
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(cm) / (cm.sum(axis=1) + 1e-12)
    return acc

# ---------------------------
# Data augmentation generator for feature space and raw windows
# ---------------------------
class FeatureAugmentor(tf.keras.utils.Sequence):
    """
    Keras Sequence to augment features on the fly:
      - gaussian noise
      - channel dropout (set some channel-features to zero)
    Input X_feats: (N, D) where D = 50 (10 channels × 5 features)
    """
    def __init__(self, X, y, batch_size=64, shuffle=True, noise_std=0.01, dropout_prob=0.05):
        self.X = X.astype(np.float32)
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        Xb = self.X[batch_idx].copy()
        yb = self.y[batch_idx]
        # gaussian noise
        Xb += np.random.normal(0, self.noise_std, size=Xb.shape).astype(np.float32)
        # channel dropout: zero out a whole channel's feature block (5 features per channel)
        num_channels = 10
        feat_per_ch = Xb.shape[1] // num_channels
        for i in range(Xb.shape[0]):
            for ch in range(num_channels):
                if np.random.rand() < self.dropout_prob:
                    start = ch * feat_per_ch
                    Xb[i, start:start+feat_per_ch] = 0.0
        return Xb, to_categorical(yb-1, num_classes=len(np.unique(self.y)))

class RawWindowAugmentor(tf.keras.utils.Sequence):
    """
    Augment raw windows (10 x 512):
     - add gaussian noise
     - randomly zero-out a channel (simulate poor electrode contact)
    """
    def __init__(self, X_raw, y, batch_size=32, shuffle=True, noise_std=0.01, channel_dropout=0.05):
        self.X = X_raw.astype(np.float32)  # shape (N, 10, 512)
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise_std = noise_std
        self.channel_dropout = channel_dropout
        self.indices = np.arange(len(X_raw))
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        Xb = self.X[batch_idx].copy()
        yb = self.y[batch_idx]
        # gaussian noise
        Xb += np.random.normal(0, self.noise_std, size=Xb.shape).astype(np.float32)
        # channel dropout
        for i in range(Xb.shape[0]):
            for ch in range(Xb.shape[1]):
                if np.random.rand() < self.channel_dropout:
                    Xb[i, ch, :] = 0.0
        return Xb, to_categorical(yb-1, num_classes=len(np.unique(self.y)))

# ---------------------------
# Model builders
# ---------------------------
def build_mlp(input_dim, num_classes, weight_decay=1e-4):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn1d(channels=10, samples=512, num_classes=8):
    inp = Input(shape=(channels, samples))
    x = Conv1D(filters=64, kernel_size=9, strides=1, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# Main training pipeline
# ---------------------------
def prepare_feature_dataset():
    # loads features for all subjects (basic only) — uses your features.py loader
    print("[DATA] loading extracted features for all subjects (this may take a moment)...")
    X_all, df_all = load_all_subjects_features(verbose=True)
    y_all = df_all['label'].astype(int).values
    subj_all = df_all['subject'].astype(int).values

    # subject-transfer split
    train_mask = np.isin(subj_all, TRAIN_SUBJECTS)
    test_mask  = np.isin(subj_all, TEST_SUBJECTS)
    X_train = X_all[train_mask]; y_train = y_all[train_mask]
    X_test  = X_all[test_mask];  y_test  = y_all[test_mask]

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # save scaler for later
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_feat.joblib"))

    print(f"[DATA] train samples: {X_train_s.shape[0]}, test samples: {X_test_s.shape[0]}")
    return X_train_s, y_train, X_test_s, y_test

def prepare_raw_dataset():
    # load raw windows for all subjects using data_loader.load_basic_subject
    Xs, ys, mids, subs = [], [], [], []
    for sid in range(1,12):
        print(f"[RAW] Loading subject {sid} raw windows...")
        X, y, mid, meta = load_basic_subject(sid)
        X, y, mid = drop_nan_labels(X, y, mid)
        Xs.append(X); ys.append(y); subs.extend([sid]*len(y))
    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    subs = np.array(subs)
    # split subject-transfer
    train_mask = np.isin(subs, TRAIN_SUBJECTS)
    test_mask  = np.isin(subs, TEST_SUBJECTS)
    X_train = X_all[train_mask]; y_train = y_all[train_mask]
    X_test  = X_all[test_mask];  y_test  = y_all[test_mask]
    print(f"[RAW] train: {X_train.shape}, test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def compute_class_weight(y):
    # simple inverse frequency
    counts = Counter(y)
    classes = sorted(list(counts.keys()))
    total = len(y)
    class_weight = {}
    for c in classes:
        class_weight[c-1] = total / (len(classes) * counts[c])  # Keras expects zero-based keys for class indices in some versions
    print("[INFO] class weights:", class_weight)
    return class_weight

def run_mlp_experiment(epochs=50, batch_size=128, use_augmentation=True):
    X_train, y_train, X_test, y_test = prepare_feature_dataset()
    num_classes = len(np.unique(y_train))

    # class weights
    cw = compute_class_weight(y_train)

    # to_categorical
    y_train_oh = to_categorical(y_train - 1, num_classes)
    y_test_oh  = to_categorical(y_test - 1, num_classes)

    model = build_mlp(X_train.shape[1], num_classes)
    model.summary()

    # callbacks
    ckpt_path = os.path.join(MODELS_DIR, "ann_mlp_best.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
    ]

    if use_augmentation:
        train_gen = FeatureAugmentor(X_train, y_train, batch_size=batch_size, noise_std=0.01, dropout_prob=0.05)
        val_split = int(0.8 * len(X_train))
        # We will use generator but still provide validation_data as a fixed split
        history = model.fit(train_gen,
                            epochs=epochs,
                            validation_data=(X_train_s[val_split:], to_categorical(y_train[val_split:]-1, num_classes)),
                            callbacks=callbacks,
                            class_weight=cw,
                            verbose=2)
    else:
        history = model.fit(X_train, y_train_oh,
                            epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, callbacks=callbacks,
                            class_weight=cw, verbose=2)

    # save final model and scaler already saved earlier
    model.save(os.path.join(MODELS_DIR, "ann_mlp_final.keras"))
    plot_training_history(history, name_prefix="ann_mlp")
    # evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1) + 1
    acc = accuracy_score(y_test, y_pred)
    print("[RESULT] MLP test acc:", acc)
    print(classification_report(y_test, y_pred, digits=4, target_names=[LABEL_MAP[i] for i in LABEL_ORDER]))

    cm = plot_confusion(y_test, y_pred, labels=LABEL_ORDER, title="MLP Confusion Matrix", outname="mlp_cm.png")
    acc_per_class = per_class_accuracy(cm)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar([LABEL_MAP[l] for l in LABEL_ORDER], acc_per_class)
    ax.set_ylim(0,1)
    ax.set_title("Per-class accuracy (MLP)")
    plt.xticks(rotation=25)
    save_fig(fig, "mlp_per_class_acc.png")
    plt.show()
    return model, acc, cm

def run_cnn_experiment(epochs=60, batch_size=64):
    X_train_raw, y_train, X_test_raw, y_test = prepare_raw_dataset()
    # Keras expects (batch, time, channels) or (batch, channels, time) depending on layers; our Conv1D uses (channels, samples)
    # ensure shape is (N, channels, samples)
    # y to categorical
    num_classes = len(np.unique(y_train))
    y_train_oh = to_categorical(y_train - 1, num_classes)
    y_test_oh  = to_categorical(y_test - 1, num_classes)

    model = build_cnn1d(channels=X_train_raw.shape[1], samples=X_train_raw.shape[2], num_classes=num_classes)
    model.summary()
    ckpt_path = os.path.join(MODELS_DIR, "cnn1d_best.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
    ]

    train_gen = RawWindowAugmentor(X_train_raw, y_train, batch_size=batch_size, noise_std=0.01, channel_dropout=0.05)
    val_split = int(0.8 * X_train_raw.shape[0])
    # We'll use validation_slice from the non-augmented validation set
    history = model.fit(train_gen, epochs=epochs,
                        validation_data=(X_train_raw[val_split:], to_categorical(y_train[val_split:]-1, num_classes)),
                        callbacks=callbacks, verbose=2)

    model.save(os.path.join(MODELS_DIR, "cnn1d_final.keras"))
    plot_training_history(history, name_prefix="cnn1d")
    # evaluate
    y_pred_prob = model.predict(X_test_raw)
    y_pred = np.argmax(y_pred_prob, axis=1) + 1
    acc = accuracy_score(y_test, y_pred)
    print("[RESULT] CNN test acc:", acc)
    print(classification_report(y_test, y_pred, digits=4, target_names=[LABEL_MAP[i] for i in LABEL_ORDER]))

    cm = plot_confusion(y_test, y_pred, labels=LABEL_ORDER, title="CNN Confusion Matrix", outname="cnn_cm.png")
    acc_per_class = per_class_accuracy(cm)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar([LABEL_MAP[l] for l in LABEL_ORDER], acc_per_class)
    ax.set_ylim(0,1)
    ax.set_title("Per-class accuracy (CNN)")
    plt.xticks(rotation=25)
    save_fig(fig, "cnn_per_class_acc.png")
    plt.show()
    return model, acc, cm

# ---------------------------
# Runner
# ---------------------------
if __name__ == "__main__":
    print("=== ANN advanced experiment ===")
    # 1) MLP on features
    try:
        model_mlp, acc_mlp, cm_mlp = run_mlp_experiment(epochs=40, batch_size=128, use_augmentation=False)
    except Exception as e:
        print("MLP experiment failed:", e)

    # 2) Optional: CNN on raw windows (comment/uncomment as needed)
    try:
        model_cnn, acc_cnn, cm_cnn = run_cnn_experiment(epochs=40, batch_size=64)
    except Exception as e:
        print("CNN experiment failed or skipped:", e)

    # 3) Save a small summary CSV
    summary = {
        "model": ["MLP", "CNN"],
        "accuracy": [float(acc_mlp) if 'acc_mlp' in locals() else None, float(acc_cnn) if 'acc_cnn' in locals() else None]
    }
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_BASE, "summary_ann_results.csv"), index=False)
    print("[DONE] Saved summary_ann_results.csv")
