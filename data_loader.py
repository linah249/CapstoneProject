import os
import pandas as pd
import numpy as np

def load_basic_subject(subject_id):
    """
    Load all 3 hand orientations (r1, r2, r3) for a given subject.

    Returns:
        X_all   -> EMG signals (num_windows, 10, 512)
        y_all   -> labels for each window (num_windows,)
        mid_all -> movement repetition ID (num_windows,)
        meta_all -> metadata for each window: filename + orientation
    """
    base_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "sEMG dataset", 
        "Basic Movements"
    )

    sbj_pattern = f"Nearlab_sbj{subject_id}_WL512_S128"

    X_all, y_all, mid_all, meta_all = [], [], [], []

    # Loop through all CSV files of the subject
    for file in os.listdir(base_path):
        if sbj_pattern in file and file.endswith(".csv"):
            file_path = os.path.join(base_path, file)
            df = pd.read_csv(file_path, header=None)

            # Extract signals and labels
            signals = df.iloc[:, :5120].values     # 10 * 512 samples
            labels = df.iloc[:, 5120].values       # gesture label (1..8)
            mids   = df.iloc[:, 5121].values       # movement ID (1..40)

            # Reshape to (num_windows, 10, 512)
            signals = signals.reshape(-1, 10, 512)

            # Extract r1 / r2 / r3 orientation from filename
            orientation = file.split("_")[-1].replace(".csv", "")

            # Append data
            X_all.append(signals)
            y_all.append(labels)
            mid_all.append(mids)
            # repeat metadata for each window
            meta_all.extend([{ "filename": file, "orientation": orientation }] * len(labels))

    # Concatenate everything
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    mid_all = np.concatenate(mid_all, axis=0)

    return X_all, y_all, mid_all, meta_all


def drop_nan_labels(X, y, mid):
    """
    Safety check: some subjects may contain NaNs in labels.
    We remove those rows.
    """
    mask = ~np.isnan(y)
    return X[mask], y[mask], mid[mask]

# --- Logic to load windows iteratively per subject ---
def stream_subject_windows_flat(subject_id, base_path):
    # This will load data file-by-file for the subject and yield flat windows
    # base_path is the directory containing your subject CSVs
    
    sbj_pattern = f"Nearlab_sbj{subject_id}_WL512_S128"
    
    # Locate the subject's files (r1, r2, r3, etc.)
    for file in os.listdir(base_path):
        if sbj_pattern in file and file.endswith(".csv"):
            file_path = os.path.join(base_path, file)
            # Read the file directly into a DataFrame
            df = pd.read_csv(file_path, header=None)
            
            # Use only EMG columns (5120 columns)
            signals_flat = df.iloc[:, :5120].values # (N_windows, 5120)
            
            # Yield data in smaller batches or chunks if necessary, but starting by yielding the file's content
            for i in range(signals_flat.shape[0]):
                 # Yields one row (one flat window) at a time
                 yield signals_flat[i, :]

# =====================================================
# 🧩 NEW: Combo Movements Loader
# =====================================================

def load_all_combo(
    folder_path=r"C:\Users\linah\Desktop\Final Year Project Linah\sEMG dataset\Combo Movements",
    window_size=512,
    step=128,
    num_combos=6,
    num_channels=10
):
    """
    Load all combo movement CSVs and create sliding windows.
    Automatically handles extra columns at the end.
    Returns:
        X -> numpy array, shape: (num_windows, 512, 10)
        y -> numpy array, labels 0..num_combos-1
    """
    X_list = []
    y_list = []

    paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    if len(paths) == 0:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    for path in paths:
        print(f"Loading combo movement: {os.path.basename(path)}")
        df = pd.read_csv(path, header=None)
        data = df.values

        # Check there are enough columns for EMG channels
        if data.shape[1] < window_size * num_channels:
            raise ValueError(f"CSV {path} has too few columns for {num_channels} channels")

        # Only use EMG columns, ignore extra label/ID columns
        emg_data = data[:, :window_size * num_channels]

        # Reshape: (num_windows, window_size, num_channels)
        signals = emg_data.reshape(-1, window_size, num_channels)

        # Split evenly for each combo class
        total_windows = signals.shape[0]
        windows_per_combo = total_windows // num_combos

        for idx in range(num_combos):
            start = idx * windows_per_combo
            end = start + windows_per_combo
            X_list.append(signals[start:end])
            y_list.extend([idx] * windows_per_combo)

    # Concatenate all
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)

    print(f"\n[INFO] Total windows created: {X.shape[0]}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[INFO] Unique combo labels: {np.unique(y)}\n")
    return X, y




