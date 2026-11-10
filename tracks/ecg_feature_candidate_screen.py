"""Screen a small, pre-declared set of complementary ECG feature candidates.

This is an exploratory development-only comparison on the verified local cache.
Each candidate is first added alone to the supplied 11-feature baseline.  Only
candidates that improve both pooled macro-F1 and mean fold macro-F1 are then
combined with the previously selected low-template-correlation fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from ecg_cinc2017 import ECGCinC2017Track, _rpeaks
from rr_candidate_comparison import (
    BASELINE_FEATURE_NAMES,
    compact_report,
    comparison_model,
)
from rr_interval_prototype import (
    low_template_correlation_fraction,
    rr_intervals_from_peaks,
)


CANDIDATE_FEATURE_NAMES = [
    "rr_histogram_entropy",
    "compensatory_pause_fraction",
    "rr_alternation_fraction",
    "qrs_amplitude_mad_ratio",
    "pre_qrs_template_correlation",
    "normalized_spectral_entropy",
]


def _valid_rr_mask(rr: np.ndarray) -> np.ndarray:
    return np.isfinite(rr) & (rr >= 0.30) & (rr <= 2.00)


def rr_histogram_entropy(rr_intervals_s, min_intervals: int = 8) -> float:
    """Normalized entropy of a fixed 50 ms RR histogram over 0.30-2.00 s."""
    rr = np.asarray(rr_intervals_s, dtype=float)
    valid = rr[_valid_rr_mask(rr)]
    if len(valid) < min_intervals:
        return float("nan")
    edges = np.arange(0.30, 2.00 + 0.0501, 0.05)
    counts, _ = np.histogram(valid, bins=edges)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(counts)))


def compensatory_pause_fraction(rr_intervals_s, min_intervals: int = 8) -> float:
    """Fraction of adjacent RR pairs resembling premature beat + compensation."""
    rr = np.asarray(rr_intervals_s, dtype=float)
    valid = _valid_rr_mask(rr)
    if int(np.count_nonzero(valid)) < min_intervals:
        return float("nan")
    median_rr = float(np.median(rr[valid]))
    adjacent = valid[:-1] & valid[1:]
    if int(np.count_nonzero(adjacent)) < min_intervals - 1:
        return float("nan")
    first = rr[:-1][adjacent]
    second = rr[1:][adjacent]
    short_then_long = (first < 0.80 * median_rr) & (second > 1.20 * median_rr)
    compensated_sum = np.abs((first + second) - 2.0 * median_rr) <= 0.40 * median_rr
    return float(np.mean(short_then_long & compensated_sum))


def rr_alternation_fraction(rr_intervals_s, min_intervals: int = 8) -> float:
    """Fraction of valid RR triples with meaningful opposite-signed changes."""
    rr = np.asarray(rr_intervals_s, dtype=float)
    valid = _valid_rr_mask(rr)
    if int(np.count_nonzero(valid)) < min_intervals:
        return float("nan")
    median_rr = float(np.median(rr[valid]))
    triples = valid[:-2] & valid[1:-1] & valid[2:]
    if int(np.count_nonzero(triples)) < min_intervals - 2:
        return float("nan")
    first_change = (rr[1:-1] - rr[:-2])[triples]
    second_change = (rr[2:] - rr[1:-1])[triples]
    meaningful = (np.abs(first_change) >= 0.10 * median_rr) & (
        np.abs(second_change) >= 0.10 * median_rr
    )
    alternates = first_change * second_change < 0
    return float(np.mean(meaningful & alternates))


def qrs_amplitude_mad_ratio(ecg, peaks, fs, min_beats: int = 5) -> float:
    """Robust variability of absolute QRS peak amplitude around a local baseline."""
    signal = np.asarray(ecg, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    half_window = max(1, int(round(0.20 * fs)))
    amplitudes = []
    for peak in peaks:
        start, stop = peak - half_window, peak + half_window + 1
        if start < 0 or stop > len(signal):
            continue
        window = signal[start:stop]
        if np.all(np.isfinite(window)):
            amplitudes.append(abs(float(signal[peak]) - float(np.median(window))))
    amplitudes = np.asarray(amplitudes, dtype=float)
    amplitudes = amplitudes[np.isfinite(amplitudes) & (amplitudes > 0)]
    if len(amplitudes) < min_beats:
        return float("nan")
    center = float(np.median(amplitudes))
    return float(np.median(np.abs(amplitudes - center)) / center)


def pre_qrs_template_correlation(ecg, peaks, fs, min_beats: int = 5) -> float:
    """Median consistency of the 250-to-60 ms region before each detected QRS."""
    signal = np.asarray(ecg, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    far = int(round(0.25 * fs))
    near = int(round(0.06 * fs))
    segments = []
    for peak in peaks:
        start, stop = peak - far, peak - near
        if start < 0 or stop > len(signal) or stop <= start:
            continue
        segment = signal[start:stop]
        if np.all(np.isfinite(segment)):
            segments.append(segment)
    if len(segments) < min_beats:
        return float("nan")
    matrix = np.asarray(segments, dtype=float)
    template = np.median(matrix, axis=0)
    template -= np.mean(template)
    template_norm = float(np.linalg.norm(template))
    if template_norm <= 0:
        return float("nan")
    correlations = []
    for segment in matrix:
        centered = segment - np.mean(segment)
        denominator = float(np.linalg.norm(centered)) * template_norm
        if denominator > 0:
            correlations.append(float(np.dot(centered, template) / denominator))
    if len(correlations) < min_beats:
        return float("nan")
    return float(np.median(correlations))


def normalized_spectral_entropy(ecg, fs) -> float:
    """Normalized Welch-PSD entropy in the fixed 0.5-40 Hz ECG band."""
    signal = np.asarray(ecg, dtype=float)
    signal = signal[np.isfinite(signal)]
    if len(signal) < max(32, int(round(2.0 * fs))):
        return float("nan")
    frequencies, power = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    band = (frequencies >= 0.5) & (frequencies <= 40.0) & np.isfinite(power)
    power = np.maximum(power[band], 0.0)
    if len(power) < 2 or float(np.sum(power)) <= 0:
        return float("nan")
    probabilities = power / np.sum(power)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(power)))


def candidate_features(record):
    ecg = np.asarray(record.epochs["ecg"][0], dtype=float)
    fs = float(record.fs)
    peaks = _rpeaks(ecg, fs)
    rr = rr_intervals_from_peaks(peaks, fs)
    return np.asarray(
        [
            rr_histogram_entropy(rr),
            compensatory_pause_fraction(rr),
            rr_alternation_fraction(rr),
            qrs_amplitude_mad_ratio(ecg, peaks, fs),
            pre_qrs_template_correlation(ecg, peaks, fs),
            normalized_spectral_entropy(ecg, fs),
        ],
        dtype=float,
    ), low_template_correlation_fraction(ecg, peaks, fs)


def build_matrices(records, track):
    baseline_rows, candidate_rows, low_rows, labels, groups = [], [], [], [], []
    for record in records:
        baseline_x, y, group = track.extract_features(record)
        candidates, low_fraction = candidate_features(record)
        baseline_rows.append(np.asarray(baseline_x[0], dtype=float))
        candidate_rows.append(candidates)
        low_rows.append(low_fraction)
        labels.append(str(y[0]))
        groups.append(str(group))
    return (
        np.vstack(baseline_rows),
        np.vstack(candidate_rows),
        np.asarray(low_rows, dtype=float),
        np.asarray(labels),
        np.asarray(groups),
    )


def evaluate(track, matrix, labels, groups, seed):
    report = track.evaluate(
        matrix,
        labels,
        groups,
        clf=comparison_model(seed),
        cfg={"seed": seed, "select": "none"},
    )
    return compact_report(report)


def plot_distributions(path, labels, candidate_x):
    classes = ["N", "A", "O", "~"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for column, (axis, name) in enumerate(zip(axes.ravel(), CANDIDATE_FEATURE_NAMES)):
        values_by_class = []
        for label in classes:
            values = candidate_x[labels == label, column]
            values_by_class.append(values[np.isfinite(values)])
        axis.boxplot(values_by_class, tick_labels=classes, showfliers=False)
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("ECG class")
        axis.set_ylabel("feature value")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Pre-declared candidate feature distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path, default=repo_root / "data_cache" / "ecg_cinc2017"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "outputs" / "feature_family_screen"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    track = ECGCinC2017Track(cfg={"preprocess": "none", "seed": args.seed})
    records = track.load(str(args.cache_dir))
    if not records:
        raise RuntimeError(f"No ECG records found in {args.cache_dir}")

    holdout_ids = set(getattr(__import__("ecg_cinc2017"), "HOLDOUT_RECORDS", ()))
    overlap = sorted({record.group for record in records} & holdout_ids)
    if overlap:
        raise RuntimeError(f"Hold-out records found in development cache: {overlap}")

    baseline_x, candidate_x, low_x, labels, groups = build_matrices(records, track)
    results = {"baseline_11": evaluate(track, baseline_x, labels, groups, args.seed)}
    results["baseline_plus_low_template_correlation_fraction"] = evaluate(
        track, np.column_stack([baseline_x, low_x]), labels, groups, args.seed
    )
    for index, feature_name in enumerate(CANDIDATE_FEATURE_NAMES):
        results[f"baseline_plus_{feature_name}"] = evaluate(
            track,
            np.column_stack([baseline_x, candidate_x[:, index]]),
            labels,
            groups,
            args.seed,
        )

    baseline = results["baseline_11"]
    passed_screen = []
    for index, feature_name in enumerate(CANDIDATE_FEATURE_NAMES):
        result = results[f"baseline_plus_{feature_name}"]
        if (
            result["pooled_macro_f1"] > baseline["pooled_macro_f1"]
            and result["fold_macro_f1_mean"] > baseline["fold_macro_f1_mean"]
        ):
            passed_screen.append(feature_name)
            results[f"baseline_plus_low_fraction_and_{feature_name}"] = evaluate(
                track,
                np.column_stack([baseline_x, low_x, candidate_x[:, index]]),
                labels,
                groups,
                args.seed,
            )

    missing_counts = {
        name: int(np.count_nonzero(~np.isfinite(candidate_x[:, index])))
        for index, name in enumerate(CANDIDATE_FEATURE_NAMES)
    }
    output = {
        "status": "exploratory_raw_as_preprocessed",
        "n_records": len(records),
        "seed": args.seed,
        "split": "record-level GroupKFold",
        "holdout_overlap": overlap,
        "baseline_feature_names": BASELINE_FEATURE_NAMES,
        "candidate_feature_names": CANDIDATE_FEATURE_NAMES,
        "candidate_missing_counts": missing_counts,
        "screen_rule": "pooled macro-F1 and mean fold macro-F1 both exceed baseline",
        "passed_individual_screen": passed_screen,
        "comparisons": results,
    }
    (args.output_dir / "comparison_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "candidate_feature_values.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["record", "class", *CANDIDATE_FEATURE_NAMES])
        for group, label, row in zip(groups, labels, candidate_x):
            writer.writerow([group, label, *row.tolist()])
    plot_distributions(
        args.output_dir / "candidate_feature_distributions.png", labels, candidate_x
    )

    print(f"Records: {len(records)} | holdout overlap: {overlap}")
    print(f"Missing values: {missing_counts}")
    print(f"Passed individual screen: {passed_screen}")
    for name, result in results.items():
        print(
            f"{name}: macro-F1={result['pooled_macro_f1']:.3f}, "
            f"kappa={result['cohens_kappa']:.3f}, "
            f"fold={result['fold_macro_f1_mean']:.3f} "
            f"+/- {result['fold_macro_f1_std']:.3f}"
        )


if __name__ == "__main__":
    main()
