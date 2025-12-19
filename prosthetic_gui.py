"""
prosthetic_gui.py - FINAL POLISHED VERSION
Features:
- True Label is BLANK until run (Clean Look).
- Full Screen Layout Fix.
- About Tab Professional & Organized.
- Graph in Load Dataset Tab.
"""

import sys, os, traceback
import numpy as np
import pandas as pd
import joblib
import time 
from functools import partial
from collections import Counter

# PyQt5 GUI
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTabWidget, QTextEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QSpinBox, QSizePolicy, QFrame, QSpacerItem
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QMovie

# Plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# TensorFlow/Keras
try:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except Exception as e:
    load_model = None
    print("Warning: TensorFlow not available.", e)

# ---------------------------
# CONFIGURATION
# ---------------------------
BASE_DIR = r"C:\Users\linah\Desktop\Final Year Project Linah"

MODEL_PATHS = {
    "ann_mlp": os.path.join(BASE_DIR, r"results\ann_advanced\models\ann_mlp_final.keras"),
    "cnn": os.path.join(BASE_DIR, r"results\ann_advanced\models\cnn1d_final.keras"),
    "ann_basic": os.path.join(BASE_DIR, r"code\results_basic_ann\basic_ann_model.h5"),
    "lstm": os.path.join(BASE_DIR, r"results\lstm_allsubjects\lstm_final.keras"),
    "gru": os.path.join(BASE_DIR, r"results\gru_allsubjects\gru_final.keras"),
    "scaler_feat": os.path.join(BASE_DIR, r"results\ann_advanced\models\scaler_feat.joblib"),
    "scaler_raw_emg": os.path.join(BASE_DIR, r"results\ann_advanced\models\scaler_raw_emg.joblib")
}

BASELINE_DIR = os.path.join(BASE_DIR, r"results\baseline_results")

LABEL_MAP = {
    1: "Flexion", 2: "Extension", 3: "Supination", 4: "Pronation",
    5: "Open Hand", 6: "Pinch", 7: "Lateral Pinch", 8: "Fist"
}

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def mean_absolute_value(x): return float(np.mean(np.abs(x)))
def root_mean_square(x): return float(np.sqrt(np.mean(x.astype(np.float64)**2)))
def waveform_length(x): return float(np.sum(np.abs(np.diff(x.astype(np.float64)))))
def zero_crossings(x, thresh=0.01):
    x = x.astype(np.float64); zc = 0
    for i in range(len(x)-1):
        if (x[i] * x[i+1] < 0) and (abs(x[i] - x[i+1]) > thresh): zc += 1
    return int(zc)
def slope_sign_changes(x, thresh=0.01):
    x = x.astype(np.float64); diff = np.diff(x); ssc = 0
    for i in range(len(diff)-1):
        if (diff[i] * diff[i+1] < 0) and (abs(diff[i] - diff[i+1]) > thresh): ssc += 1
    return int(ssc)

def extract_features_simple(window):
    feats = []
    for ch in range(window.shape[0]):
        x = window[ch].astype(np.float64)
        feats += [mean_absolute_value(x), root_mean_square(x), waveform_length(x), zero_crossings(x), slope_sign_changes(x)]
    return np.array(feats, dtype=np.float32)

EXTERNAL_FEATURE_FN = None
try:
    from features import extract_features_from_raw
    EXTERNAL_FEATURE_FN = extract_features_from_raw
except Exception:
    pass

def prepare_feature_vector(window_10x512):
    if EXTERNAL_FEATURE_FN is not None:
        try:
            out = EXTERNAL_FEATURE_FN(np.expand_dims(window_10x512, axis=0))
            if isinstance(out, np.ndarray):
                return out.reshape(1, -1).astype(np.float32)
        except Exception as e:
            print(f"External extractor failed ({e}), using simple fallback.")
    return extract_features_simple(window_10x512).reshape(1, -1)

class ModelsContainer:
    def __init__(self):
        self.models = {}; self.scaler = None; self.raw_scaler = None; self.baselines = {}
    def load_tf(self, key, path):
        if not load_model or not os.path.exists(path): return False
        try: self.models[key] = load_model(path); return True
        except: return False
    def load_scaler(self, key, path):
        if not os.path.exists(path): return False
        try:
            s = joblib.load(path)
            if key == "scaler_feat": self.scaler = s
            else: self.raw_scaler = s
            return True
        except: return False
    def auto_load_baselines(self):
        candidates = {
            "RandomForest": os.path.join(BASELINE_DIR, "RandomForest", "RandomForest.joblib"),
            "SVM_RBF": os.path.join(BASELINE_DIR, "SVM_RBF", "SVM_RBF.joblib"),
            "kNN": os.path.join(BASELINE_DIR, "kNN", "kNN.joblib")
        }
        for k, p in candidates.items():
            if os.path.exists(p):
                try: self.baselines[k] = joblib.load(p)
                except: pass

models = ModelsContainer()

def parse_emg_csv_path(filepath, selected_index=0):
    df = pd.read_csv(filepath, header=None)
    arr = df.values
    if selected_index >= len(arr): selected_index = 0
    row = arr[selected_index]
    label = None
    if len(row) >= 5122:
        emg = row[:5120].astype(np.float32)
        try: label = int(row[5120])
        except: label = None
    elif len(row) == 5120:
        emg = row.astype(np.float32)
    else:
        if row.size >= 5120: emg = row.flatten()[:5120].astype(np.float32)
        else: raise ValueError("Invalid row length")
    return emg.reshape(10, 512), label, df

def get_rnn_input_shape(model):
    try: return model.input_shape[1:] if model.input_shape else None
    except: return None

def predict_all_models(window_10x512, container):
    results = {}
    feats = prepare_feature_vector(window_10x512)
    raw_in = window_10x512.copy().astype(np.float32)
    if container.raw_scaler:
        try: raw_in = container.raw_scaler.transform(raw_in.reshape(1, -1)).reshape(10, 512)
        except: pass
    
    raw_cnn = raw_in.reshape(1, 10, 512)
    raw_rnn = np.transpose(raw_in).reshape(1, 512, 10) 

    # Baselines
    for name, mdl in container.baselines.items():
        try: 
            if hasattr(mdl, "predict_proba"):
                probs = mdl.predict_proba(feats)
                pred = int(np.argmax(probs)) + 1
                results[f"Baseline: {name}"] = (pred, probs.ravel())
            else:
                pred = int(mdl.predict(feats)[0])
                results[f"Baseline: {name}"] = (pred, None)
        except Exception as e: results[f"Baseline: {name}"] = (None, str(e))

    # DL Models
    dl_config = [("ann_basic", feats, False), ("ann_mlp", feats, True), ("cnn", raw_cnn, False), ("lstm", raw_rnn, False), ("gru", raw_rnn, False)]
    model_names = {"ann_basic":"ANN Basic", "ann_mlp":"ANN Advanced", "cnn":"CNN", "lstm":"LSTM", "gru":"GRU"}

    for key, inp, use_s in dl_config:
        if key in container.models:
            try:
                m = container.models[key]
                if use_s and container.scaler: inp = container.scaler.transform(inp)
                if key in ["lstm", "gru"]:
                    ishape = get_rnn_input_shape(m)
                    if ishape and ishape == (10, 512): inp = raw_in.reshape(1, 10, 512)
                probs = m.predict(inp, verbose=0)
                pred = int(np.argmax(probs)) + 1
                results[model_names[key]] = (pred, probs.ravel())
            except Exception as e: results[model_names[key]] = (None, str(e))
    return results

# ---------------------------
# MAIN GUI CLASS
# ---------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Prosthetic Hand Control System")
        self.resize(1300, 850) 
        
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Set a larger title for the main window
        self.main_title = QLabel("AI Prosthetic Hand Control System")
        self.main_title.setProperty("style", "header")
        self.main_title.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.main_title)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.tab_home = QWidget(); self.tab_data = QWidget()
        self.tab_compare = QWidget(); self.tab_predict = QWidget()
        self.tab_about = QWidget()

        self.tabs.addTab(self.tab_home, "Home / Overview")
        self.tabs.addTab(self.tab_data, "Load Dataset")
        self.tabs.addTab(self.tab_predict, "Prediction Demo")
        self.tabs.addTab(self.tab_about, "About / Credits")

        self.loaded_window = None; self.loaded_df = None
        self.current_filepath = None; self.true_label = None
        self.selected_row = 0; self.last_results = {}

        self.emg_fig = Figure(figsize=(8, 4), dpi=100)
        self.emg_canvas = FigureCanvas(self.emg_fig)
        self.prob_fig = Figure(figsize=(5, 4), dpi=100)
        self.prob_canvas = FigureCanvas(self.prob_fig)
        self.comp_fig = Figure(figsize=(6, 4), dpi=100)
        self.comp_canvas = FigureCanvas(self.comp_fig)

        self._build_home_tab()
        self._build_data_tab()
        self._build_compare_tab()
        self._build_predict_tab()
        self._build_about_tab()

        footer = QHBoxLayout()
        self.status_label = QLabel("System Ready. Please load models.")
        footer.addWidget(self.status_label)
        self.btn_load = QPushButton("Load Models Now")
        self.btn_load.clicked.connect(self.load_models)
        footer.addWidget(self.btn_load)
        self.main_layout.addLayout(footer)

        QtCore.QTimer.singleShot(200, self.load_models)

    def load_models(self):
        self.status_label.setText("Loading models...")
        QApplication.processEvents()
        c = 0
        for k, p in MODEL_PATHS.items():
            if "scaler" in k: c += models.load_scaler(k, p)
            else: c += models.load_tf(k, p)
        models.auto_load_baselines()
        self.status_label.setText(f"Models Loaded. ({c} components ready)")

    # --- HOME TAB ---
    def _build_home_tab(self):
        outer = QVBoxLayout()
        outer.setAlignment(Qt.AlignCenter)
        container = QWidget(); container.setFixedWidth(950)
        layout = QVBoxLayout(container); layout.setSpacing(30)

        sub = QLabel("Advanced Deep Learning Classification\n(ANN • CNN • LSTM • GRU)")
        sub.setStyleSheet("font-size: 16pt; color: #aaa; font-weight: bold;")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        self.anim_label = QLabel()
        self.anim_label.setAlignment(Qt.AlignCenter)
        gif_path = os.path.join(BASE_DIR, "code", "hand_animation.gif")
        if os.path.exists(gif_path):
            self.movie = QMovie(gif_path)
            self.movie.setScaledSize(QSize(500, 350))
            self.anim_label.setMovie(self.movie)
            self.movie.start()
        else:
            self.anim_label.setText("[ Animation File Not Found ]")
            self.anim_label.setStyleSheet("border: 2px dashed #555; padding: 50px;")
        layout.addWidget(self.anim_label)

        guide = QLabel(
            "<b>Quick Start Guide:</b><br><br>"
            "1. <b>Load Dataset:</b> Open a CSV file to see EMG signals.<br>"
            "2. <b>Prediction Demo:</b> Select an index and Run Models to see classification.<br>"
            "3. <b>Comparison:</b> Analyze model accuracy charts."
        )
        guide.setStyleSheet("background-color: #2b2b2b; padding: 25px; border-radius: 12px; font-size: 12pt; border: 1px solid #444;")
        guide.setAlignment(Qt.AlignCenter)
        layout.addWidget(guide)

        outer.addWidget(container)
        self.tab_home.setLayout(outer)

    # --- DATASET TAB ---
    def _build_data_tab(self):
        layout = QVBoxLayout()
        h = QHBoxLayout()
        btn = QPushButton("Select CSV File")
        btn.setFixedWidth(200)
        btn.clicked.connect(self._browse_file)
        h.addWidget(btn)
        self.lbl_file_info = QLabel("No file loaded.")
        h.addWidget(self.lbl_file_info)
        layout.addLayout(h)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Window Index:"))
        self.spin_idx = QSpinBox()
        self.spin_idx.setRange(0, 0)
        self.spin_idx.valueChanged.connect(self._update_data_view)
        h2.addWidget(self.spin_idx)
        layout.addLayout(h2)

        layout.addWidget(self.emg_canvas, stretch=1)
        self.tab_data.setLayout(layout)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
        if path:
            self.current_filepath = path
            try:
                _, _, df = parse_emg_csv_path(path, 0)
                self.loaded_df = df
                self.spin_idx.setMaximum(len(df)-1)
                self._update_data_view(0)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _update_data_view(self, idx):
        if not self.current_filepath: return
        self.selected_row = idx
        try:
            win, label, _ = parse_emg_csv_path(self.current_filepath, idx)
            self.loaded_window = win
            self.true_label = label 
            
            self.lbl_file_info.setText(f"File: {os.path.basename(self.current_filepath)} | Rows: {len(self.loaded_df)} | Index: {idx}")
            
            self.emg_fig.clear()
            ax = self.emg_fig.add_subplot(111)
            ax.plot(win.T)
            ax.set_title(f"Raw EMG Signal - Window {idx}")
            ax.set_xlabel("Time (Samples)")
            ax.set_ylabel("Amplitude (V)")
            self.emg_fig.tight_layout()
            self.emg_canvas.draw()
            
            # Reset Prediction Tab (SHOW BLANK LABEL)
            self.lbl_pred_result_label.setText("True Label:")
            self.lbl_pred_result_label.setStyleSheet("color: #aaa; font-size: 14pt; font-weight: bold;")
            self.table_res.setRowCount(0)
            self.prob_fig.clear()
            self.prob_canvas.draw()
            
        except Exception as e:
            print("Error update view:", e)

    # --- PREDICTION TAB ---
    def _build_predict_tab(self):
        layout = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("Model Selection:"))
        self.combo_models = QComboBox()
        self.combo_models.addItems(["Run All Models", "ANN Advanced", "CNN", "LSTM", "GRU", "ANN Basic"])
        h.addWidget(self.combo_models)
        
        btn_run = QPushButton("RUN PREDICTION")
        btn_run.setStyleSheet("background-color: #00bcd4; color: black; font-weight: bold; font-size: 11pt;")
        btn_run.clicked.connect(self.run_prediction)
        h.addWidget(btn_run)
        layout.addLayout(h)

        # True Label (Blank initially - Updated logic)
        self.lbl_pred_result_label = QLabel("True Label:")
        self.lbl_pred_result_label.setStyleSheet("font-size: 14pt; font-weight: bold; margin: 15px 0; color: #aaa;")
        self.lbl_pred_result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_pred_result_label)

        h_diag = QHBoxLayout()
        self.latency_label = QLabel("Last Runtime: N/A")
        self.disagreement_label = QLabel("Model Agreement: N/A")
        h_diag.addWidget(self.latency_label)
        h_diag.addWidget(self.disagreement_label)
        layout.addLayout(h_diag)

        split = QHBoxLayout()
        self.table_res = QTableWidget()
        self.table_res.setColumnCount(3)
        self.table_res.setHorizontalHeaderLabels(["Model", "Prediction", "Confidence"])
        self.table_res.horizontalHeader().setStretchLastSection(True)
        split.addWidget(self.table_res, stretch=4)
        
        right_layout = QVBoxLayout()
        self.combo_chart = QComboBox()
        self.combo_chart.currentIndexChanged.connect(self._update_chart)
        right_layout.addWidget(QLabel("Select Model for Chart:"))
        right_layout.addWidget(self.combo_chart)
        right_layout.addWidget(self.prob_canvas)
        split.addLayout(right_layout, stretch=6)
        
        layout.addLayout(split)
        self.tab_predict.setLayout(layout)

    def run_prediction(self):
        if self.loaded_window is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first!")
            return
            
        self.status_label.setText("Running inference...")
        QApplication.processEvents()
        
        # REVEAL TRUE LABEL NOW
        lbl_str = LABEL_MAP.get(self.true_label, "Unknown") if self.true_label else "Unknown"
        self.lbl_pred_result_label.setText(f"True Label: {lbl_str} (Class {self.true_label})")
        self.lbl_pred_result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00e676;")

        t_start = time.time()
        results = predict_all_models(self.loaded_window, models)
        t_end = time.time()
        self.last_results = results
        
        self.latency_label.setText(f"Last Runtime: {(t_end - t_start)*1000:.2f} ms")
        
        preds = [v[0] for v in results.values() if isinstance(v[0], int)]
        if preds:
            common, count = Counter(preds).most_common(1)[0]
            self.disagreement_label.setText(f"Model Agreement: {count}/{len(preds)} agree on {LABEL_MAP.get(common, common)}")
        else:
            self.disagreement_label.setText("Model Agreement: N/A")

        choice = self.combo_models.currentText()
        display_keys = results.keys() if choice == "Run All Models" else [choice]
        
        self.table_res.setRowCount(0)
        self.combo_chart.clear()
        
        for name in display_keys:
            if name not in results: continue
            pred, probs = results[name]
            
            row = self.table_res.rowCount()
            self.table_res.insertRow(row)
            self.table_res.setItem(row, 0, QTableWidgetItem(name))
            
            p_text = LABEL_MAP.get(pred, str(pred))
            item_pred = QTableWidgetItem(f"{pred}: {p_text}")
            if self.true_label and pred == self.true_label:
                item_pred.setBackground(Qt.darkGreen)
                item_pred.setForeground(Qt.white)
            self.table_res.setItem(row, 1, item_pred)
            
            conf_txt = f"{np.max(probs):.2%}" if probs is not None else "N/A"
            self.table_res.setItem(row, 2, QTableWidgetItem(conf_txt))
            
            if probs is not None:
                self.combo_chart.addItem(name)

        self.status_label.setText("Inference Complete.")
        
    def _update_chart(self):
        name = self.combo_chart.currentText()
        if not name or name not in self.last_results: return
        _, probs = self.last_results[name]
        if probs is None: return
        
        self.prob_fig.clear()
        ax = self.prob_fig.add_subplot(111)
        labels = list(LABEL_MAP.values())
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, probs, align='center', color='#00bcd4')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title(f'{name} Confidence')
        ax.set_xlim(0, 1.0)
        self.prob_fig.tight_layout()
        self.prob_canvas.draw()

    # --- COMPARE TAB ---
    def _build_compare_tab(self):
        l = QVBoxLayout()
        btn = QPushButton("Load 'model_comparison_summary.csv'")
        btn.clicked.connect(self._load_comp_csv)
        l.addWidget(btn)
        l.addWidget(self.comp_canvas)
        self.tab_compare.setLayout(l)

    def _load_comp_csv(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV (*.csv)")
        if not f: return
        try:
            df = pd.read_csv(f)
            self.comp_fig.clear()
            ax = self.comp_fig.add_subplot(111)
            ax.bar(df['model'], df['accuracy'], color='#7c4dff')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Benchmarking")
            self.comp_fig.tight_layout()
            self.comp_canvas.draw()
        except: pass

    # --- ABOUT TAB (Professional & Organized) ---
    def _build_about_tab(self):
        l = QVBoxLayout()
        info = QTextEdit()
        info.setReadOnly(True)
        info.setStyleSheet("border: none; background-color: transparent; font-size: 13pt; color: #e0e0e0;")
        
        # NEW PROFESSIONAL HTML LAYOUT
        html = """
        <div style="text-align: center;">
            <h1 style='color: #00bcd4; font-size: 24pt; margin-bottom: 10px;'>Project Information</h1>
            <hr style='border: 1px solid #555; width: 60%;'>
        </div>
        <div style="padding: 20px;">
            <p style='font-size: 14pt;'><b>Project Title:</b> AI-Driven sEMG Control System for Prosthetic Hands</p>
            <p style='font-size: 14pt;'><b>Student Name:</b> Linah Adil Elshiekh Abdalla</p>
            <p style='font-size: 14pt;'><b>Supervisor Name:</b> Ts. Dr Tan Tee Hean</p>
            <p style='font-size: 14pt;'><b>Year:</b> 2025</p>
        </div>
        <table width="100%" cellspacing="20" style="margin-top: 10px;">
            <tr>
                <td width="50%" valign="top" style="background-color: #2b2b2b; padding: 15px; border-radius: 10px; border: 1px solid #3a3f47;">
                    <h2 style='color: #00bcd4; border-bottom: 1px solid #00bcd4; padding-bottom: 5px;'>System Overview</h2>
                    <ul style="line-height: 1.6;">
                        <li><b>Input Data:</b> 10-Channel sEMG (NearLab Dataset)</li>
                        <li><b>Preprocessing:</b> 512ms Windowing, Dual Scaling Strategy</li>
                        <li><b>Feature Extraction:</b> Time-Domain (MAV, RMS, WL, ZC, SSC)</li>
                        <li><b>Classification:</b> Real-time Ensemble Inference</li>
                    </ul>
                </td>
                <td width="50%" valign="top" style="background-color: #2b2b2b; padding: 15px; border-radius: 10px; border: 1px solid #3a3f47;">
                    <h2 style='color: #00bcd4; border-bottom: 1px solid #00bcd4; padding-bottom: 5px;'>AI Models Implemented</h2>
                    <ul style="line-height: 1.6;">
                        <li><b>Deep Learning:</b> LSTM, GRU, 1D-CNN, ANN (MLP)</li>
                        <li><b>Classical ML:</b> Random Forest, SVM, kNN (Baselines)</li>
                    </ul>
                </td>
            </tr>
        </table>
        <div style="text-align: center; margin-top: 30px;">
            <p style='color: #888; font-size: 12pt;'>Developed using Python, PyQt5, TensorFlow, Scikit-Learn, and Pandas.</p>
        </div>
        """
        info.setHtml(html)
        l.addWidget(info)
        self.tab_about.setLayout(l)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    css_path = os.path.join(BASE_DIR, "code", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            app.setStyleSheet(f.read())
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
