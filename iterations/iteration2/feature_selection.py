"""
Testing different methods for selecting features and their impact on the baseline. 
Selectors tester were:
    1. None (baseline to compare against)
    2. Variance (removing features with 0 variance)
    3. Anova, k = 8
    4. Anova, k = 11
    5. Tree
    6. Lasso

Features extracted are the ones in the last version of feature_extraction.py (11 original features + RR-interval entropy + RR-interval difference entropy + QRS-morphology median)
In order for Anova to handle nan value, a median imputer was added to the track  

Purpose:
    Testing performance of different selectors

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings), 1999 real recording, part of the developmental set

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

Results: 
    From the macro-F1 scores we could not see any improvement from any selector. This does not mean that every feature is usefull for classifying the data.
    When looking at the confusion matrices none of the selectors showed an improvemtn in classifying the normal, AF or "other" ecg:s.
    Tree and Anova (k=11) improved the noisy class with Tree showing the best classification of noisy. Altough Anova (k=11) did not show the same
    improvement in classifying noisy records, it performed better on the other classes than Tree. 
    Anova (k=11) seemed to improve noisy classification while not ruining the other classes at the same time. Since the noisy class is still the 
    one that is being the most missclassified i think a selector which improve the noisy class could prove beneficial (even if it comes at a cost of disturbing the other classes).

    But the main improvment in the noisy class might come from extracting better features for noisy classification

    None:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.93 | 0.00 | 0.06 | 0.01 | 1200 |
    | **A** | 0.08 | 0.70 | 0.21 | 0.01 | 173 |
    | **O** | 0.40 | 0.04 | 0.56 | 0.00 | 567 |
    | **~** | 0.42 | 0.00 | 0.14 | 0.44 | 59 |

    **macro_f1** — mean macro_f1 0.694 (sd 0.035, range 0.654-0.744 across 5 folds); pooled 0.699
    - cohens_kappa: pooled 0.593, mean over folds 0.592
    - balanced_accuracy: pooled 0.657, mean over folds 0.657
    - accuracy: pooled 0.789, mean over folds 0.789
    - worst fold: 3 (macro_f1 0.654, n=400) — read this one's errors before the mean's

    Variance:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.93 | 0.00 | 0.06 | 0.01 | 1200 |
    | **A** | 0.08 | 0.70 | 0.21 | 0.01 | 173 |
    | **O** | 0.40 | 0.04 | 0.56 | 0.00 | 567 |
    | **~** | 0.42 | 0.00 | 0.14 | 0.44 | 59 |

    **macro_f1** — mean macro_f1 0.694 (sd 0.035, range 0.654-0.744 across 5 folds); pooled 0.699
    - cohens_kappa: pooled 0.593, mean over folds 0.592
    - balanced_accuracy: pooled 0.657, mean over folds 0.657
    - accuracy: pooled 0.789, mean over folds 0.789
    - worst fold: 3 (macro_f1 0.654, n=400) — read this one's errors before the mean's

    Anova (k=8):
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.90 | 0.01 | 0.08 | 0.01 | 1200 |
    | **A** | 0.08 | 0.69 | 0.23 | 0.00 | 173 |
    | **O** | 0.42 | 0.05 | 0.52 | 0.01 | 567 |
    | **~** | 0.37 | 0.03 | 0.14 | 0.46 | 59 |

    **macro_f1** — mean macro_f1 0.669 (sd 0.045, range 0.604-0.709 across 5 folds); pooled 0.673
    - cohens_kappa: pooled 0.548, mean over folds 0.547
    - balanced_accuracy: pooled 0.644, mean over folds 0.642
    - accuracy: pooled 0.764, mean over folds 0.764
    - worst fold: 1 (macro_f1 0.604, n=400) — read this one's errors before the mean's

    Anova (k=11):
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.92 | 0.00 | 0.07 | 0.01 | 1200 |
    | **A** | 0.08 | 0.69 | 0.23 | 0.01 | 173 |
    | **O** | 0.40 | 0.05 | 0.55 | 0.01 | 567 |
    | **~** | 0.39 | 0.02 | 0.12 | 0.47 | 59 |

    **macro_f1** — mean macro_f1 0.688 (sd 0.043, range 0.621-0.733 across 5 folds); pooled 0.693
    - cohens_kappa: pooled 0.579, mean over folds 0.578
    - balanced_accuracy: pooled 0.659, mean over folds 0.656
    - accuracy: pooled 0.780, mean over folds 0.780
    - worst fold: 1 (macro_f1 0.621, n=400) — read this one's errors before the mean's

    Tree:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.88 | 0.00 | 0.09 | 0.03 | 1200 |
    | **A** | 0.08 | 0.67 | 0.24 | 0.01 | 173 |
    | **O** | 0.36 | 0.06 | 0.56 | 0.03 | 567 |
    | **~** | 0.32 | 0.03 | 0.10 | 0.54 | 59 |

    **macro_f1** — mean macro_f1 0.656 (sd 0.039, range 0.625-0.724 across 5 folds); pooled 0.657
    - cohens_kappa: pooled 0.557, mean over folds 0.557
    - balanced_accuracy: pooled 0.664, mean over folds 0.666
    - accuracy: pooled 0.762, mean over folds 0.762
    - worst fold: 1 (macro_f1 0.625, n=400) — read this one's errors before the mean's

    Lasso:
    | true \ pred | N | A | O | ~ | n |
    |---|---|---|---|---|---|
    | **N** | 0.93 | 0.00 | 0.06 | 0.01 | 1200 |
    | **A** | 0.08 | 0.70 | 0.21 | 0.01 | 173 |
    | **O** | 0.40 | 0.04 | 0.56 | 0.00 | 567 |
    | **~** | 0.42 | 0.00 | 0.14 | 0.44 | 59 |

    **macro_f1** — mean macro_f1 0.694 (sd 0.035, range 0.654-0.744 across 5 folds); pooled 0.699
    - cohens_kappa: pooled 0.593, mean over folds 0.592
    - balanced_accuracy: pooled 0.657, mean over folds 0.657
    - accuracy: pooled 0.789, mean over folds 0.789
    - worst fold: 3 (macro_f1 0.654, n=400) — read this one's errors before the mean's

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
            #Configuration 1
            "seed" : SEED,
            "preprocess": "none",
            "r_peak":"R2",
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "Variance": {
            #Configuration 1
            "seed" : SEED,
            "preprocess": "none",
            "r_peak":"R2",
            "select": "variance",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "Anova, k=8": {
            #Configuration 1
            "seed" : SEED,
            "preprocess": "none",
            "r_peak":"R2",
            "select": "anova",
            "select_k": 8,
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "Anova, k=11": {
            #Configuration 1
            "seed" : SEED,
            "preprocess": "none",
            "r_peak":"R2",
            "select": "anova",
            "select_k": 11,
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "Tree": {
            #Configuration 1
            "seed" : SEED,
            "preprocess": "none",
            "r_peak":"R2",
            "select": "tree",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "Lasso": {
            #Configuration 1
            "seed" : SEED,
            "preprocess": "none",
            "r_peak":"R2",
            "select": "lasso",
            "imbalance": "balanced",
            "cv_max_splits": 5,
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
        print("No developmental data found inside evaluation data")
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