"""
Experiment B0 - Original baseline supplied from Farhad

Purpose:
    Extablish reference classification perfromance, this is what we compare the other iterations to

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings)

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

from ecg_cinc2017 import ECGCinC2017Track

def main():

    print("\n======================================")
    print("EXPERIMENT B0 — ORIGINAL ECG BASELINE")
    print("======================================")

    SEED = 0
    SUBSET = 2000

    USE_REAL = True

    #Find these in adapter.py, theres a dict defined as "DEFAULT_CFG"
    cfg = {
        "seed" : SEED,
        "preprocess": "none",
        "select": "none",
        "imbalance": "balanced",
        "cv_max_splits": 5,
    }

    track = ECGCinC2017Track()
    print("\nConfiguration")
    print(cfg)

    # ----------------------------------------------
    # 2. DOWNLOAD AND LOAD REAL ECG DATA
    # ----------------------------------------------

    if USE_REAL:
        cache_dir = PROJECT_ROOT / "data_cache" / "ecg_B0"
        print("\nDownloading ECG recordings...")
        track.download(str(cache_dir), subset = SUBSET, seed = SEED)
        records = track.load(str(cache_dir))
        track.assert_no_holdout(records)
    else:
        records = track.smoke()
        print(f"Generated {len(records)} synthetic ECG recordings")

    
    #This print function together with the other print functions are defined below the main function
    printDataInfo(track, records)

    # ----------------------------------------------
    # 3. PREPROCESSING + FEATURE EXTRACTION
    # ----------------------------------------------

    print("\nExtracting features...")

    X, y, groups = track.build_dataset(records, cfg = cfg)

    printFeatureInfo(X, y, groups)

    # ----------------------------------------------
    # 4. MACHINE LEARNING + CROSS-VALIDATION
    # ----------------------------------------------

    print("\nRunning 5-fold GroupKFold...")

    report = track.evaluate(X, y, groups, cfg = cfg)

    # ----------------------------------------------
    # 5. EVALUATION RESULTS
    # ----------------------------------------------

    print("\n======================================")
    print("B0 — CLASSIFICATION RESULTS")
    print("======================================")

    printEvaluationScores(track, report)
    printConfusionMatrix(track, report)
    printPerClassPerformance(track, report)


# ----------------------------------------------
# These are a bunch of print function, so print can easily be removed when just testing stuff
# ----------------------------------------------

def printDataInfo(track, records):
    print(f"Number of recordings: {len(records)}")

    print("\nClass distribution")
    labels = np.array([rec.labels[0] for rec in records])
    for cls in track.meta.classes:
        print(f"{cls}: {np.sum(labels==cls)}")


def printFeatureInfo(X, y, groups):
    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {y.shape}")
    print(f"Unique recordings: {len(set(groups))}")


def printEvaluationScores(track, report):
    y_true = report["y_true"]
    y_pred = report["y_pred"]

    classes = track.meta.classes

    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

    print(f"\nMacro-F1: {macro_f1:.4f}")
    print(f"Cohen's kappa: {report['cohens_kappa']:.4f}")

    fold_scores = []

    for fold in report["per_fold"]:
        print(
            f"Fold {fold['fold']}: "
            f"Macro-F1 = {fold['macro_f1']:.4f}"
        )

        fold_scores.append(fold["macro_f1"])

    fold_scores = np.array(fold_scores)

    print("\nMacro-F1 across folds:")

    print(f"Mean: {np.mean(fold_scores):.4f}")
    print(f"SD: {np.std(fold_scores):.4f}")
    print(f"Min: {np.min(fold_scores):.4f}")
    print(f"Max: {np.max(fold_scores):.4f}")


def printConfusionMatrix(track, report):
    y_true = report["y_true"]
    y_pred = report["y_pred"]

    classes = track.meta.classes

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    print("\nConfusion matrix:")
    print("Class order:", classes)
    print(cm)


def printPerClassPerformance(track, report):
    y_true = report["y_true"]
    y_pred = report["y_pred"]

    classes = track.meta.classes

    print("\nPer-class performance:")

    print(classification_report(y_true, y_pred, labels=classes, target_names=["Normal", "AF", "Other", "Noisy"], zero_division=0))

    print("\nExperiment B0 completed!")



if __name__ == "__main__":
    main()