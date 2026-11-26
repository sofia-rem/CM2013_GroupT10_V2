"""
Testing how 2 new features 
Purpose:
    Further testing of classifiers, trying to tune them

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings), 1999 real recording, part of the developmental set

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

Results:
    No reason to change rpeak detection method

    


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
        "Defualt (iteration 3 config)": {
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

        "new feature: short long rr intervals": {
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
            "include_short_long_rr_pair": True,
        },

        "new feature: low template correlation": {
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
            "include_low_template_correlation": True,
        },

        "both new features": {
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
            "include_short_long_rr_pair": True,
            "include_low_template_correlation": True,
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
    n_cols = 2
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