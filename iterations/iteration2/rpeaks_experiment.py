"""
Experiment script, copy this script into another file when running the experiment.
Make function changes in the baseline (AAAA_t10track.py) and fix the configurations (cfg) in this script.
Choose real or smoke data

Purpose:
    FILL IN THE PURPOSE HERE

Dataset: 
    PhysioNet/CinC Challenge 2017 (real ECG recordings), 1999 real recording, part of the developmental set

Evaluation:
    5-fold GroupKFold, split by ECG recording

Primary metric:
    Macro-F1

Notes from experiment:
    By adding adaptive thresholding to r-peak detection the detection for each class improved, but mostly for the noisy (0.31 to 0.51 and AF (0.50 to 0.57) classes.
    By adding more accurate r-peak localization and tracking missed r-peaks the noisy class got worse (from 0.51 to 0.44) but classification of AF increased (0.57 to 0.61)
    This also got a higher macro-F1 score (both mean and pooled) but at the same time had a higher sd and range.
    
    Here it's important to not only look at the macro-F1, we could say that adding more accurate r-peak localization and tracking missed peaks is better
    by looking at the macro-F1, but at the same time we have a weaker noisy class. The noisy class don't have nearly as many records as the other classes,
    so it might not impact the score as much as other classes, but it's still important to address.

    Since we use r-peak detection to detect the r-peaks, we feel that it's more sensible to have the r-peak detection method that detects AF better.
    We would like to use other features to detect the noisy class, such as ACF
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

from AAAA_t10track_ecg import ECGCinC2017Track
from report import summarize_report, plot_confusion
import matplotlib.pyplot as plt

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
        "deafult detection" : {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R0",
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "Adaptive threshold" : {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R1",
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },

        "Adaptive threshold, localisation and missing peaks detection" : {
            "seed" : SEED,
            "preprocess": "none",
            "r_peak": "R2",
            "select": "none",
            "imbalance": "balanced",
            "cv_max_splits": 5,
        },
    }

    cfg = {
        "seed" : SEED,
        "preprocess": "wavelet",
            "wavelet": "db4",
        "ecg_band": (0.5, 40),
        "select": "none",
        "imbalance": "balanced",
        "cv_max_splits": 5,
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

    
    #This print function together with the other print functions are defined below the main function
    printDataInfo(track, records)

    import matplotlib.pyplot as plt
    import numpy as np
    from AAAA_t10track_ecg import _rpeaks, _rpeaks_adaptive, _rpeaks_improved, _divisible_len
    fig, axes = plt.subplots(len(track.meta.classes), 1, figsize=(9, 6), sharex=True)

    for ax, cls in zip(axes, track.meta.classes):
        rec = next((r for r in records if r.labels[0] == cls), None)

        if rec is None: 
            ax.set_visible(False)
            continue
        filt_rec = track.preprocess(rec, cfg=cfg)
        x = filt_rec.epochs['ecg'][0]
        fs = filt_rec.fs
        n = min(len(x), int(30*fs))
        t = np.arange(n)/fs

        ax.plot(t, x[:n], lw=0.8)

        # pk_default = _rpeaks(x, fs)
        # pk_default = pk_default[pk_default < n]
        # ax.plot(pk_default/fs, x[pk_default], 'r.', ms=8)

        # pk_adaptive = _rpeaks_adaptive(x, fs)
        # pk_adaptive = pk_adaptive[pk_adaptive < n]
        # ax.plot(pk_adaptive/fs, x[pk_adaptive], 'c.', ms=8)

        pk_improved = _rpeaks_improved(x, fs)
        pk_improved = pk_improved[pk_improved < n]
        ax.plot(pk_improved/fs, x[pk_improved], 'g.', ms=8)

        pk_new = _divisible_len(x, fs)
        pk_new = pk_new[pk_new < n]
        ax.plot(pk_new/fs, x[pk_new], 'm.', ms=8)

        ax.set_ylabel(f'class {cls}')
    
    axes[-1].set_xlabel('time (s)')
    axes[0].set_title('CinC-2017 rhythm strips (red = R-peaks)')


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