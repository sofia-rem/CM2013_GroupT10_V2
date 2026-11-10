"""Compare seven candidate ECG features against the supplied baseline.

For this preliminary experiment, each cached ECG is treated as if it were the
preprocessed signal.  The script does not modify the main track, does not access
or download hold-out records, and uses the same record-level folds for every
comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_cinc2017 import ECGCinC2017Track, _rpeaks
from rr_interval_prototype import (
    beat_template_correlation,
    low_template_correlation_fraction,
    low_template_correlation_fraction_leave_one_out,
    low_template_correlation_severity_leave_one_out,
    normalized_rdr_occupancy,
    normalized_rr_mad,
    rr_intervals_from_peaks,
    rr_sample_entropy,
)


BASELINE_FEATURE_NAMES = [
    "heart_rate_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50_percent",
    "rr_coefficient_of_variation",
    "rr_outside_0.6_to_1.2_s_fraction",
    "rr_interval_count",
    "qrs_to_high_frequency_power_ratio",
    "powerline_49_to_51_hz_power",
    "ecg_standard_deviation",
    "ecg_kurtosis",
]

CANDIDATE_FEATURE_NAMES = [
    "normalized_rr_mad",
    "rr_sample_entropy",
    "beat_template_correlation",
    "low_template_correlation_fraction",
    "low_template_correlation_fraction_leave_one_out",
    "low_template_correlation_severity_leave_one_out",
    "normalized_rdr_occupancy",
]


def candidate_features(record) -> np.ndarray:
    """Extract the seven candidates from one assumed-preprocessed Recording."""
    ecg = np.asarray(record.epochs["ecg"][0], dtype=float)
    fs = float(record.fs)
    peaks = _rpeaks(ecg, fs)
    rr = rr_intervals_from_peaks(peaks, fs)
    return np.array(
        [
            normalized_rr_mad(rr),
            rr_sample_entropy(rr),
            beat_template_correlation(ecg, peaks, fs),
            low_template_correlation_fraction(ecg, peaks, fs),
            low_template_correlation_fraction_leave_one_out(ecg, peaks, fs),
            low_template_correlation_severity_leave_one_out(ecg, peaks, fs),
            normalized_rdr_occupancy(rr),
        ],
        dtype=float,
    )


def build_feature_matrices(records, track):
    """Build baseline and candidate matrices without learned transformations."""
    baseline_rows, candidate_rows, labels, groups = [], [], [], []
    for record in records:
        baseline_x, y, group = track.extract_features(record)
        baseline_rows.append(np.asarray(baseline_x[0], dtype=float))
        candidate_rows.append(candidate_features(record))
        labels.append(str(y[0]))
        groups.append(str(group))
    return (
        np.vstack(baseline_rows),
        np.vstack(candidate_rows),
        np.asarray(labels),
        np.asarray(groups),
    )


def comparison_model(seed: int) -> Pipeline:
    """Supplied baseline model plus fold-local candidate-feature imputation."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def class_recalls(report: dict) -> dict[str, float]:
    confusion = np.asarray(report["confusion"], dtype=float)
    denominators = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion),
        denominators,
        out=np.full(len(denominators), np.nan),
        where=denominators > 0,
    )
    return {
        str(label): float(value)
        for label, value in zip(report["labels"], recalls)
    }


def compact_report(report: dict) -> dict:
    spread = report["spread"]["macro_f1"]
    return {
        "pooled_macro_f1": float(report["macro_f1"]),
        "cohens_kappa": float(report["cohens_kappa"]),
        "fold_macro_f1_mean": float(spread["mean"]),
        "fold_macro_f1_std": float(spread["std"]),
        "fold_macro_f1_min": float(spread["min"]),
        "fold_macro_f1_max": float(spread["max"]),
        "per_class_recall": class_recalls(report),
        "confusion_matrix": np.asarray(report["confusion"]).tolist(),
    }


def write_feature_csv(path: Path, groups, labels, candidate_x) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record", "class", *CANDIDATE_FEATURE_NAMES])
        for group, label, row in zip(groups, labels, candidate_x):
            writer.writerow([group, label, *row.tolist()])


def plot_distributions(path: Path, labels, candidate_x) -> None:
    classes = ["N", "A", "O", "~"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()
    for column, (axis, name) in enumerate(zip(axes, CANDIDATE_FEATURE_NAMES)):
        values_by_class = []
        for label in classes:
            values = candidate_x[labels == label, column]
            values_by_class.append(values[np.isfinite(values)])
        axis.boxplot(values_by_class, tick_labels=classes, showfliers=False)
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("ECG class")
        axis.set_ylabel("feature value")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[len(CANDIDATE_FEATURE_NAMES) :]:
        axis.set_visible(False)
    fig.suptitle("Candidate feature distributions (raw ECG treated as preprocessed)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "data_cache" / "ecg_cinc2017",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "rr_candidate_comparison",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    track = ECGCinC2017Track(cfg={"preprocess": "none", "seed": args.seed})
    records = track.load(str(args.cache_dir))
    if not records:
        raise RuntimeError(f"No ECG records found in {args.cache_dir}")

    # Use a published hold-out list when the local upstream version provides it.
    # This script never downloads data, and the existing 344-record cache was
    # previously checked against the teacher's list with zero matches.
    holdout_ids = set(getattr(__import__("ecg_cinc2017"), "HOLDOUT_RECORDS", ()))
    overlap = sorted({record.group for record in records} & holdout_ids)
    if overlap:
        raise RuntimeError(f"Hold-out records found in development cache: {overlap}")

    baseline_x, candidate_x, labels, groups = build_feature_matrices(records, track)
    feature_sets = {
        "baseline_11": baseline_x,
        "baseline_plus_rr_mad": np.column_stack([baseline_x, candidate_x[:, 0]]),
        "baseline_plus_rr_sample_entropy": np.column_stack(
            [baseline_x, candidate_x[:, 1]]
        ),
        "baseline_plus_beat_template_correlation": np.column_stack(
            [baseline_x, candidate_x[:, 2]]
        ),
        "baseline_plus_low_template_correlation_fraction": np.column_stack(
            [baseline_x, candidate_x[:, 3]]
        ),
        "baseline_plus_low_fraction_leave_one_out": np.column_stack(
            [baseline_x, candidate_x[:, 4]]
        ),
        "baseline_plus_low_severity_leave_one_out": np.column_stack(
            [baseline_x, candidate_x[:, 5]]
        ),
        "baseline_plus_normalized_rdr_occupancy": np.column_stack(
            [baseline_x, candidate_x[:, 6]]
        ),
        "baseline_plus_low_fraction_and_rdr_occupancy": np.column_stack(
            [baseline_x, candidate_x[:, 3], candidate_x[:, 6]]
        ),
        "baseline_plus_all_previous_three": np.column_stack(
            [baseline_x, candidate_x[:, :3]]
        ),
    }

    results = {}
    for name, matrix in feature_sets.items():
        report = track.evaluate(
            matrix,
            labels,
            groups,
            clf=comparison_model(args.seed),
            cfg={"seed": args.seed, "select": "none"},
        )
        results[name] = compact_report(report)

    missing_counts = {
        name: int(np.count_nonzero(~np.isfinite(candidate_x[:, index])))
        for index, name in enumerate(CANDIDATE_FEATURE_NAMES)
    }
    output = {
        "status": "preliminary_raw_as_preprocessed",
        "n_records": len(records),
        "seed": args.seed,
        "split": "record-level GroupKFold",
        "holdout_overlap": overlap,
        "baseline_feature_names": BASELINE_FEATURE_NAMES,
        "candidate_feature_names": CANDIDATE_FEATURE_NAMES,
        "candidate_missing_counts": missing_counts,
        "comparisons": results,
    }

    write_feature_csv(
        args.output_dir / "candidate_feature_values.csv",
        groups,
        labels,
        candidate_x,
    )
    plot_distributions(
        args.output_dir / "candidate_feature_distributions.png",
        labels,
        candidate_x,
    )
    result_path = args.output_dir / "comparison_results.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Records: {len(records)}")
    print(f"Missing candidate values: {missing_counts}")
    for name, result in results.items():
        print(
            f"{name}: pooled macro-F1={result['pooled_macro_f1']:.3f}, "
            f"kappa={result['cohens_kappa']:.3f}, "
            f"fold macro-F1={result['fold_macro_f1_mean']:.3f} "
            f"+/- {result['fold_macro_f1_std']:.3f}"
        )
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
