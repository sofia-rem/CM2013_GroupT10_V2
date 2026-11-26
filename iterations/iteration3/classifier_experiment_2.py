"""
Test different classifiers perfromance on baseline, classifiers tested are:
    1. kNN
    2. SVM
    3. Random forest

Purpose:
    Further testing of classifiers, trying to tune them

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings), 1999 real recording, part of the developmental set

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

Results:

    Comments:
    This is part 2 of classifier experimentation, check part 1 first
    Setting the min samples per leaf to 5 increased the classification for noisy records to an all time high (0.76), while still maintaining 
    a lot of the classification for Normal ecg:s. Overall the Random forest with min leaf samples of 5 has the best balance between classes and a great 
    macro-F1

    Random forest: max depth = 5
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.79 | 0.02 | 0.16 | 0.03 | 1200 |
    | **A** | 0.02 | 0.83 | 0.14 | 0.01 | 173 |
    | **O** | 0.25 | 0.14 | 0.56 | 0.05 | 567 |
    | **~** | 0.15 | 0.03 | 0.12 | 0.69 | 59 |

    **macro_f1** — mean macro_f1 0.637 (sd 0.039, range 0.587-0.681 across 5 folds); pooled 0.639
    - cohens_kappa: pooled 0.523, mean over folds 0.523
    - balanced_accuracy: pooled 0.718, mean over folds 0.719
    - accuracy: pooled 0.723, mean over folds 0.723
    - worst fold: 1 (macro_f1 0.587, n=400) — read this one's errors before the mean's

    Random forest: max depth = 10
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.89 | 0.01 | 0.09 | 0.01 | 1200 |
    | **A** | 0.06 | 0.72 | 0.21 | 0.01 | 173 |
    | **O** | 0.34 | 0.06 | 0.59 | 0.01 | 567 |
    | **~** | 0.31 | 0.02 | 0.15 | 0.53 | 59 |

    **macro_f1** — mean macro_f1 0.691 (sd 0.021, range 0.663-0.718 across 5 folds); pooled 0.696
    - cohens_kappa: pooled 0.592, mean over folds 0.591
    - balanced_accuracy: pooled 0.683, mean over folds 0.681
    - accuracy: pooled 0.781, mean over folds 0.781
    - worst fold: 1 (macro_f1 0.663, n=400) — read this one's errors before the mean's

    Random forest: Min samples per leaf = 5
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.85 | 0.01 | 0.10 | 0.04 | 1200 |
    | **A** | 0.03 | 0.79 | 0.16 | 0.01 | 173 |
    | **O** | 0.29 | 0.09 | 0.57 | 0.05 | 567 |
    | **~** | 0.14 | 0.02 | 0.08 | 0.76 | 59 |

    **macro_f1** — mean macro_f1 0.674 (sd 0.034, range 0.623-0.715 across 5 folds); pooled 0.675
    - cohens_kappa: pooled 0.581, mean over folds 0.580
    - balanced_accuracy: pooled 0.745, mean over folds 0.749
    - accuracy: pooled 0.764, mean over folds 0.764
    - worst fold: 1 (macro_f1 0.623, n=400) — read this one's errors before the mean's

    3 Nearest Neighbour:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.85 | 0.01 | 0.12 | 0.01 | 1200 |
    | **A** | 0.10 | 0.68 | 0.20 | 0.02 | 173 |
    | **O** | 0.38 | 0.08 | 0.52 | 0.01 | 567 |
    | **~** | 0.39 | 0.02 | 0.12 | 0.47 | 59 |

    **macro_f1** — mean macro_f1 0.637 (sd 0.029, range 0.599-0.678 across 5 folds); pooled 0.638
    - cohens_kappa: pooled 0.503, mean over folds 0.502
    - balanced_accuracy: pooled 0.633, mean over folds 0.635
    - accuracy: pooled 0.733, mean over folds 0.733
    - worst fold: 0 (macro_f1 0.599, n=400) — read this one's errors before the mean's

    10 Nearest Neighbour:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.90 | 0.01 | 0.09 | 0.01 | 1200 |
    | **A** | 0.07 | 0.73 | 0.20 | 0.01 | 173 |
    | **O** | 0.40 | 0.07 | 0.51 | 0.01 | 567 |
    | **~** | 0.42 | 0.07 | 0.05 | 0.46 | 59 |

    **macro_f1** — mean macro_f1 0.665 (sd 0.052, range 0.606-0.723 across 5 folds); pooled 0.668
    - cohens_kappa: pooled 0.547, mean over folds 0.546
    - balanced_accuracy: pooled 0.650, mean over folds 0.651
    - accuracy: pooled 0.761, mean over folds 0.761
    - worst fold: 1 (macro_f1 0.606, n=400) — read this one's errors before the mean's

    SVM: gamma = auto
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.78 | 0.03 | 0.13 | 0.06 | 1200 |
    | **A** | 0.01 | 0.86 | 0.12 | 0.01 | 173 |
    | **O** | 0.26 | 0.13 | 0.57 | 0.04 | 567 |
    | **~** | 0.10 | 0.07 | 0.14 | 0.69 | 59 |

    **macro_f1** — mean macro_f1 0.632 (sd 0.023, range 0.610-0.665 across 5 folds); pooled 0.633
    - cohens_kappa: pooled 0.534, mean over folds 0.533
    - balanced_accuracy: pooled 0.728, mean over folds 0.730
    - accuracy: pooled 0.727, mean over folds 0.727
    - worst fold: 0 (macro_f1 0.610, n=400) — read this one's errors before the mean's
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

from AAAA_t10track_ecg import ECGCinC2017Track
from report import summarize_report, plot_confusion
import matplotlib.pyplot as plt
import math
def main():

    print("\n======================================")
    print("ENTER EXPERIMENT NAME — ENTER PURPOSE")
    print("======================================")

    SEED = 0
    DEV_SUBSET = 2000
    EVAL_SUBSET = 2000

    USE_REAL = True
    #Find these in adapter.py, theres a dict defined as "DEFAULT_CFG"

    configs = {
        "None": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "Random forest, max depth = 5": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "max_depth": 5,
        },

        "Random forest, max depth = 10": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "max_depth": 10,
        },

        "Random forest, min sample leaf = 5": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "rf",
            "n_estimators" : 200,
            "min_samples_leaf": 5,
        },

        "3 Nearest Neighbours (distance)": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "knn",
            "n_neighbors": 3,
            "weights": "distance"
        },

        "10 Nearest Neighbours (distance)": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "knn",
            "n_neighbors": 10,
            "weights": "distance"
        },

        "SVM (RBF, C=1)": {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
            "classifier": "svm",
            "kernel": "rbf",
            "C":1,
            "gamma": "auto"
        },
    }


    track = ECGCinC2017Track()

    # ----------------------------------------------
    # 2. DOWNLOAD AND LOAD REAL ECG DATA
    # ----------------------------------------------
    if USE_REAL:
        cache_dir_dev = PROJECT_ROOT / "data_cache" / "dev"
        cache_dir_eval = PROJECT_ROOT / "data_cache" / "eval"

        print("\nDownloading ECG recordings for development...")
        track.download(str(cache_dir_dev), subset = DEV_SUBSET, seed = SEED)
        dev_records = track.load(str(cache_dir_dev))
        dev_ids = {rec.group for rec in dev_records}
        track.assert_no_holdout(dev_records)

        print("\nDownloading ECG recordings for evaluation...")
        track.download(str(cache_dir_eval), subset = EVAL_SUBSET, seed = SEED, exclude_ids=dev_ids)
        eval_records = track.load(str(cache_dir_eval))
        eval_ids = {rec.group for rec in eval_records}

        if not dev_ids.isdisjoint(eval_ids):
            raise ValueError("Rerords found in both develepment set and evaluation set")
        print("Developmental and evaluation records as disjoint")
        
        track.assert_no_holdout(eval_records)
        records = dev_records

    else:
        records = track.smoke()
        print(f"Generated {len(records)} synthetic ECG recordings")

    printDataInfo(track, records)
    
    n_configs = len(configs)
    n_cols = 3
    n_rows = math.ceil(n_configs / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for i, (name, cfg) in enumerate(configs.items()):
        print("\nConfiguration")
        print(cfg)
        current_track = ECGCinC2017Track()
        print("\nExtracting features...")
        X, y, groups = current_track.build_dataset(records, cfg=cfg)
        report = current_track.evaluate(X, y, groups, cfg=cfg)
        current_track.report(report)
        plot_confusion(
            report["y_true"],
            report["y_pred"],
            labels=track.meta.classes,
            title=name,
            ax=axes[i]
        )

    for ax in axes[len(configs):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()



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





if __name__ == "__main__":
    main()