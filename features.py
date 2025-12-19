# features.py — Final Version with Advanced TD Features, Data Loading, and Plotting Fix

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from scipy import signal
from statsmodels.regression.linear_model import yule_walker # For AR coefficients
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE # <--- Added for better utility, though PCA is used here
from sklearn.ensemble import RandomForestClassifier

# Assuming data_loader.py is accessible in your working directory
from data_loader import load_basic_subject, drop_nan_labels 


# --------------------------
# Visualization: publication style defaults (UNCHANGED)
# --------------------------
plt.rcParams.update({
    "figure.figsize": (8,6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif"
})

# --------------------------
# Feature Functions (per 1D signal)
# --------------------------
def mean_absolute_value(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))

def root_mean_square(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64)**2)))

def waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x.astype(np.float64)))))

def zero_crossings(x: np.ndarray, thresh: float = 0.01) -> int:
    x = x.astype(np.float64)
    zc = 0
    for i in range(len(x)-1):
        if (x[i] * x[i+1] < 0) and (abs(x[i] - x[i+1]) > thresh):
            zc += 1
    return int(zc)

def slope_sign_changes(x: np.ndarray, thresh: float = 0.01) -> int:
    x = x.astype(np.float64)
    diff = np.diff(x)
    ssc = 0
    for i in range(len(diff)-1):
        if (diff[i] * diff[i+1] < 0) and (abs(diff[i] - diff[i+1]) > thresh):
            ssc += 1
    return int(ssc)

# --- ADVANCED TD FEATURES ---

def integrated_emg(x: np.ndarray) -> float:
    """IEMG: Sum of absolute values."""
    return float(np.sum(np.abs(x.astype(np.float64))))

def variance(x: np.ndarray) -> float:
    """VAR: Variance of the signal."""
    return float(np.var(x.astype(np.float64)))

def autoregressive_coefficients(x: np.ndarray, order: int = 4) -> List[float]:
    """AR Coefficients (Yule-Walker method) - returns 4 coefficients."""
    try:
        ar_coeffs, _ = yule_walker(x.astype(np.float64), order, method='ywm')
        return ar_coeffs.tolist()
    except Exception:
        return [0.0] * order

# --------------------------
# Per-window feature extraction 
# --------------------------
def extract_features_from_window(window: np.ndarray, fs: int = 2048) -> np.ndarray:
    """
    Returns: 1D numpy array of length 110 dimensions.
    """
    if window.ndim != 2:
        raise ValueError("window must be 2D array (channels, samples)")

    ch, samples = window.shape
    feats = []
    for c in range(ch):
        x = window[c, :].astype(np.float64)
        
        feats.append(mean_absolute_value(x))
        feats.append(root_mean_square(x))
        feats.append(waveform_length(x))
        feats.append(zero_crossings(x, thresh=0.01))
        feats.append(slope_sign_changes(x, thresh=0.01))
        
        feats.append(integrated_emg(x))
        feats.append(variance(x))
        feats.extend(autoregressive_coefficients(x, order=4)) 
        
    return np.array(feats, dtype=np.float32)

# --------------------------
# Batch extraction (many windows)
# --------------------------
def extract_features_batch(X: np.ndarray,
                           y: Optional[np.ndarray] = None,
                           channel_names: Optional[List[str]] = None,
                           sample_for_plot: int = 1000
                           ) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    X: (N, channels=10, samples=512)
    Returns: X_feats (N, 110)
    """
    N = X.shape[0]
    ch_count = X.shape[1]
    
    if channel_names is None:
        channel_names = [f"CH{c+1}" for c in range(ch_count)]
        
    base_feat_names = ["MAV", "RMS", "WL", "ZC", "SSC", "IEMG", "VAR", "AR1", "AR2", "AR3", "AR4"]
    feat_names = []
    for ch in channel_names:
        feat_names += [f"{ch}_{f}" for f in base_feat_names]

    Xf = np.zeros((N, len(feat_names)), dtype=np.float32)
    for i in range(N):
        Xf[i, :] = extract_features_from_window(X[i, :])

    df = pd.DataFrame(Xf, columns=feat_names)
    if y is not None:
        df['label'] = y.astype(int)

    return Xf, df


def extract_features_from_raw(X_raw):
    """ Used as the EXTERNAL_FEATURE_FN fallback in the GUI. """
    Xf, _ = extract_features_batch(X_raw) 
    return Xf

# --------------------------
# Plotting Utility (FIXED: The missing function)
# --------------------------

# Define a standard LABEL_MAP for plotting simplicity
LABEL_MAP = {1: "Flexion", 2: "Extension", 3: "Supination", 4: "Pronation", 5: "Open Hand", 6: "Pinch", 7: "Lateral Pinch", 8: "Fist"}

def plot_pca_embedding(X: np.ndarray, y: np.ndarray, title: str = "Feature Embedding (PCA)", n_components: int = 2):
    """
    Performs PCA dimensionality reduction and plots the feature space based on labels.
    Required by train_ann_advanced.py for visualization.
    """
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plotting logic
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        mask = y == label
        label_name = LABEL_MAP.get(label, f"Class {label}")
        ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label=label_name, alpha=0.6)
        
    # Set plot labels
    ax.set_title(title)
    ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend()
    plt.show()


# --------------------------
# High-Level Feature Loading and Combining (FIXED)
# --------------------------

def load_all_subjects_features(subject_ids: range = range(1, 12),
                               include_labels: bool = True,
                               verbose: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    CRITICAL FUNCTION: Loads raw EMG data for specified subjects, extracts 110-D features, 
    and combines them into a single dataset matrix (for ANN/Baselines training).
    """
    X_all_list = []
    y_all_list = []
    subj_all_list = []

    for sid in subject_ids:
        if verbose:
            print(f"[Loader] Processing Subject {sid}...")
        
        # NOTE: load_basic_subject comes from the imported data_loader.py
        X_raw, y, mid, _ = load_basic_subject(sid)
        X_raw, y, mid = drop_nan_labels(X_raw, y, mid) # Clean data
        
        # Extract features using the new 110D batch extractor
        X_feats, df_feats = extract_features_batch(X_raw, y=y)
        
        X_all_list.append(X_feats)
        y_all_list.append(y)
        subj_all_list.extend([sid] * len(y))

    # Combine all feature data
    X_all = np.vstack(X_all_list)
    y_all = np.concatenate(y_all_list)
    
    df_final = pd.DataFrame(X_all, columns=df_feats.columns[:-1]) 
    
    if include_labels:
        df_final['label'] = y_all
    
    df_final['subject'] = np.array(subj_all_list)
    
    if verbose:
        print(f"[Loader] Finished loading all subjects. Total samples: {X_all.shape[0]}")
        print(f"[Loader] Final feature matrix shape: {X_all.shape}")
        
    return X_all, df_final