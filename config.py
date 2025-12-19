# config.py
import os

ROOT = os.path.dirname(os.path.dirname(__file__))  # parent of code/
DATA_DIR = os.path.join(ROOT, "sEMG dataset")
BASIC_DIR = os.path.join(DATA_DIR, "Basic Movements")
COMBO_DIR = os.path.join(DATA_DIR, "Combo Movements")

RESULTS_DIR = os.path.join(ROOT, "results")
ANN_RESULTS = os.path.join(RESULTS_DIR, "ann_advanced")
COMBO_RESULTS = os.path.join(RESULTS_DIR, "combo_results")
BASELINE_RESULTS = os.path.join(RESULTS_DIR, "baselines")

MODELS_DIR = os.path.join(ANN_RESULTS, "models")
FIGS_DIR = os.path.join(ANN_RESULTS, "figs")

# make sure folders exist
for p in [RESULTS_DIR, ANN_RESULTS, COMBO_RESULTS, BASELINE_RESULTS, MODELS_DIR, FIGS_DIR]:
    os.makedirs(p, exist_ok=True)

# Dataset settings
TRAIN_SUBJECTS = list(range(1, 9))
TEST_SUBJECTS = [9, 10, 11]

RANDOM_SEED = 42

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



