"""Compare ECG preprocessing alternatives on the local non-holdout development set.

This keeps the teacher's feature extractor, classifier, record-level GroupKFold,
and random seed fixed. Only the preprocessing configuration changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tracks"))

from ecg_cinc2017 import ECGCinC2017Track, HOLDOUT_RECORDS  # noqa: E402


CONFIGS = {
    "baseline_none": {
        "preprocess": "none",
        "seed": 0,
    },
    "sofia_wavelet_notch_savgol": {
        "preprocess": "denoise",
        "baseline": "wavelet",
        "wavelet": "sym4",
        "level": 5,
        "powerline": "notch",
        "f0": 50.0,
        "q": 40.0,
        "harmonics": (1, 2),
        "broadband": "savgol",
        "savgol_order": 2,
        "win": 9,
        "seed": 0,
    },
    "classmate_poly3_savgol": {
        # Assumption from the chat description: cubic polynomial baseline
        # detrending plus Savitzky-Golay denoising. The teacher's denoise()
        # executes baseline removal before broadband smoothing.
        "preprocess": "denoise",
        "baseline": "detrend",
        "poly_order": 3,
        "broadband": "savgol",
        "savgol_order": 2,
        "win": 9,
        "seed": 0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"G:\CM2013_GroupT10\data_cache\ecg_cinc2017"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "preprocessing_comparison",
    )
    return parser.parse_args()


def summarise(rep, seconds):
    confusion = np.asarray(rep["confusion"], dtype=int)
    totals = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion), totals, out=np.zeros_like(totals, dtype=float), where=totals > 0
    )
    macro_spread = rep["spread"]["macro_f1"]
    return {
        "pooled_macro_f1": float(rep["macro_f1"]),
        "cohens_kappa": float(rep["cohens_kappa"]),
        "balanced_accuracy": float(rep["balanced_accuracy"]),
        "accuracy": float(rep["accuracy"]),
        "fold_macro_f1_mean": float(macro_spread["mean"]),
        "fold_macro_f1_std": float(macro_spread["std"]),
        "labels": [str(value) for value in rep["labels"]],
        "per_class_recall": {
            str(label): float(value) for label, value in zip(rep["labels"], recall)
        },
        "confusion": confusion.tolist(),
        "runtime_seconds": float(seconds),
    }


def plot_confusions(results, output_path):
    fig, axes = plt.subplots(1, len(results), figsize=(13.5, 4.2), constrained_layout=True)
    for axis, (name, result) in zip(axes, results.items()):
        matrix = np.asarray(result["confusion"])
        axis.imshow(matrix, cmap="Blues")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
        labels = result["labels"]
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            name.replace("_", " ")
            + f"\nF1={result['pooled_macro_f1']:.3f}, κ={result['cohens_kappa']:.3f}"
        )
    fig.suptitle("Real non-holdout ECG preprocessing comparison")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = ECGCinC2017Track(cfg=CONFIGS["baseline_none"])
    records = loader.load(str(args.data_dir))
    if not records:
        raise RuntimeError(f"No ECG records found in {args.data_dir}")

    overlap = sorted({record.group for record in records} & set(HOLDOUT_RECORDS))
    if overlap:
        raise RuntimeError(f"Hold-out records found in development cache: {overlap}")

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"Running {name} on {len(records)} records...", flush=True)
        started = time.perf_counter()
        track = ECGCinC2017Track(cfg=cfg)
        X, y, groups = track.build_dataset(records)
        rep = track.evaluate(X, y, groups)
        results[name] = summarise(rep, time.perf_counter() - started)
        result = results[name]
        print(
            f"  macro-F1={result['pooled_macro_f1']:.3f}, "
            f"kappa={result['cohens_kappa']:.3f}, "
            f"fold={result['fold_macro_f1_mean']:.3f} +/- "
            f"{result['fold_macro_f1_std']:.3f}",
            flush=True,
        )

    output = {
        "status": "real_non_holdout_development_comparison",
        "n_records": len(records),
        "class_counts": {
            label: sum(str(record.labels[0]) == label for record in records)
            for label in ("N", "A", "O", "~")
        },
        "holdout_overlap": overlap,
        "split": "record-level 5-fold GroupKFold",
        "fixed_components": "teacher feature extractor and default balanced classifier",
        "classmate_interpretation": (
            "baseline='detrend', poly_order=3, broadband='savgol', "
            "savgol_order=2, win=9; teacher denoise order applies"
        ),
        "configs": CONFIGS,
        "results": results,
    }
    (args.output_dir / "comparison_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    plot_confusions(results, args.output_dir / "confusion_matrices.png")
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
