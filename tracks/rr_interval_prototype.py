"""Prototype RR-interval feature for the ECG capstone.

This module is intentionally independent of the team's preprocessing pipeline.
It can be tested with artificial RR intervals now and connected to Sofia's
preprocessed ECG (or R-peak indices) later.

This is a candidate feature prototype, not a final experiment result.
"""

from __future__ import annotations

import numpy as np


def rr_intervals_from_peaks(
    r_peak_samples: np.ndarray | list[float],
    sampling_rate_hz: float,
) -> np.ndarray:
    """Convert R-peak sample positions into RR intervals measured in seconds."""
    peaks = np.asarray(r_peak_samples, dtype=float)

    if peaks.ndim != 1:
        raise ValueError("r_peak_samples must be a one-dimensional sequence")
    if not np.all(np.isfinite(peaks)):
        raise ValueError("r_peak_samples must contain only finite values")
    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be a positive finite number")
    if len(peaks) < 2:
        return np.array([], dtype=float)
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("r_peak_samples must be strictly increasing")

    return np.diff(peaks) / sampling_rate_hz


def normalized_rr_mad(
    rr_intervals_s: np.ndarray | list[float],
    min_intervals: int = 3,
) -> float:
    """Return median absolute RR deviation divided by the median RR interval.

    A larger value means that the beat-to-beat intervals vary more.  NaN is
    returned when there are too few valid intervals; the final pipeline must
    report and handle those missing values explicitly rather than replacing
    them silently with zero.
    """
    rr = np.asarray(rr_intervals_s, dtype=float)

    if rr.ndim != 1:
        raise ValueError("rr_intervals_s must be a one-dimensional sequence")
    if min_intervals < 1:
        raise ValueError("min_intervals must be at least 1")

    valid_rr = rr[np.isfinite(rr) & (rr > 0)]
    if len(valid_rr) < min_intervals:
        return float("nan")

    median_rr = float(np.median(valid_rr))
    rr_mad = float(np.median(np.abs(valid_rr - median_rr)))
    return rr_mad / median_rr


def rr_sample_entropy(
    rr_intervals_s: np.ndarray | list[float],
    embedding_dimension: int = 2,
    tolerance_fraction: float = 0.2,
    min_intervals: int = 10,
) -> float:
    """Estimate sample entropy of an RR sequence.

    The tolerance is ``tolerance_fraction * standard deviation`` of this one
    recording.  NaN means that the record is too short/constant or that no
    template matches were available; it must be imputed inside a training fold.
    """
    rr = np.asarray(rr_intervals_s, dtype=float)
    valid_rr = rr[np.isfinite(rr) & (rr > 0)]
    m = int(embedding_dimension)

    if m < 1:
        raise ValueError("embedding_dimension must be at least 1")
    if tolerance_fraction <= 0:
        raise ValueError("tolerance_fraction must be positive")
    if len(valid_rr) < max(min_intervals, m + 2):
        return float("nan")

    tolerance = tolerance_fraction * float(np.std(valid_rr))
    if not np.isfinite(tolerance) or tolerance <= 0:
        return float("nan")

    def matching_pairs(template_length: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(
            valid_rr, template_length
        )
        count = 0
        for i in range(len(templates) - 1):
            distances = np.max(
                np.abs(templates[i + 1 :] - templates[i]), axis=1
            )
            count += int(np.count_nonzero(distances <= tolerance))
        return count

    matches_m = matching_pairs(m)
    matches_m_plus_one = matching_pairs(m + 1)
    if matches_m == 0 or matches_m_plus_one == 0:
        return float("nan")

    return float(-np.log(matches_m_plus_one / matches_m))


def normalized_rdr_occupancy(
    rr_intervals_s: np.ndarray | list[float],
    grid_s: float = 0.025,
    min_rr_s: float = 0.30,
    max_rr_s: float = 2.00,
    min_intervals: int = 8,
) -> float:
    """Return normalized occupancy of the RR-versus-delta-RR grid.

    Each valid point is ``(RR[i], RR[i] - RR[i-1])``.  Both consecutive RR
    intervals must lie in the fixed physiological range so that removing an
    invalid interval never creates an artificial adjacency.  The plane is
    quantized into fixed ``grid_s`` cells, and the number of occupied cells is
    divided by the number of valid points.  A larger value indicates a more
    dispersed, less repetitive RR transition pattern.

    This is a short-record adaptation inspired by the RdR-map method of Lian et
    al.; it is not claimed to reproduce their fixed 32/64/128-interval windows.
    """
    rr = np.asarray(rr_intervals_s, dtype=float)
    if rr.ndim != 1:
        raise ValueError("rr_intervals_s must be a one-dimensional sequence")
    if not np.isfinite(grid_s) or grid_s <= 0:
        raise ValueError("grid_s must be a positive finite number")
    if not np.isfinite(min_rr_s) or not np.isfinite(max_rr_s):
        raise ValueError("RR bounds must be finite")
    if min_rr_s <= 0 or max_rr_s <= min_rr_s:
        raise ValueError("RR bounds must satisfy 0 < min_rr_s < max_rr_s")
    if min_intervals < 2:
        raise ValueError("min_intervals must be at least 2")

    valid = np.isfinite(rr) & (rr >= min_rr_s) & (rr <= max_rr_s)
    if int(np.count_nonzero(valid)) < min_intervals:
        return float("nan")

    adjacent_valid = valid[:-1] & valid[1:]
    if int(np.count_nonzero(adjacent_valid)) < min_intervals - 1:
        return float("nan")

    current_rr = rr[1:][adjacent_valid]
    delta_rr = np.diff(rr)[adjacent_valid]
    rr_cells = np.floor(current_rr / grid_s).astype(np.int64)
    delta_cells = np.floor(delta_rr / grid_s).astype(np.int64)
    occupied_cells = len(set(zip(rr_cells.tolist(), delta_cells.tolist())))
    return float(occupied_cells / len(current_rr))


def _beat_template_correlations(
    ecg: np.ndarray | list[float],
    r_peak_samples: np.ndarray | list[float],
    sampling_rate_hz: float,
    before_peak_s: float = 0.20,
    after_peak_s: float = 0.40,
    min_beats: int = 3,
    leave_one_out: bool = False,
) -> np.ndarray:
    """Return each aligned beat's correlation with a robust median template.

    With ``leave_one_out=True``, the beat being scored is excluded from the
    template so that it cannot make its own match look artificially better.
    """
    signal = np.asarray(ecg, dtype=float)
    peaks = np.asarray(r_peak_samples, dtype=int)

    if signal.ndim != 1 or peaks.ndim != 1:
        raise ValueError("ecg and r_peak_samples must be one-dimensional")
    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be a positive finite number")

    before = int(round(before_peak_s * sampling_rate_hz))
    after = int(round(after_peak_s * sampling_rate_hz))
    beats = []
    for peak in peaks:
        start, stop = int(peak) - before, int(peak) + after
        if start < 0 or stop > len(signal):
            continue
        beat = signal[start:stop]
        if len(beat) == before + after and np.all(np.isfinite(beat)):
            beats.append(beat)

    if len(beats) < min_beats:
        return np.array([], dtype=float)

    beat_matrix = np.asarray(beats, dtype=float)
    correlations = []
    shared_template = None
    if not leave_one_out:
        shared_template = np.median(beat_matrix, axis=0)

    for index, beat in enumerate(beat_matrix):
        template = (
            np.median(np.delete(beat_matrix, index, axis=0), axis=0)
            if leave_one_out
            else shared_template
        )
        template = template - np.mean(template)
        template_norm = float(np.linalg.norm(template))
        if template_norm <= 0:
            continue
        centered = beat - np.mean(beat)
        denominator = float(np.linalg.norm(centered)) * template_norm
        if denominator > 0:
            correlations.append(float(np.dot(centered, template) / denominator))

    if len(correlations) < min_beats:
        return np.array([], dtype=float)
    return np.asarray(correlations, dtype=float)


def beat_template_correlation(
    ecg: np.ndarray | list[float],
    r_peak_samples: np.ndarray | list[float],
    sampling_rate_hz: float,
    before_peak_s: float = 0.20,
    after_peak_s: float = 0.40,
    min_beats: int = 3,
) -> float:
    """Return the median beat-to-template correlation for one ECG record."""
    correlations = _beat_template_correlations(
        ecg,
        r_peak_samples,
        sampling_rate_hz,
        before_peak_s=before_peak_s,
        after_peak_s=after_peak_s,
        min_beats=min_beats,
    )
    if len(correlations) < min_beats:
        return float("nan")
    return float(np.median(correlations))


def low_template_correlation_fraction(
    ecg: np.ndarray | list[float],
    r_peak_samples: np.ndarray | list[float],
    sampling_rate_hz: float,
    correlation_threshold: float = 0.50,
    before_peak_s: float = 0.20,
    after_peak_s: float = 0.40,
    min_beats: int = 3,
) -> float:
    """Return the fraction of valid beats whose template correlation is low.

    The threshold is fixed before evaluation rather than tuned on the full
    development cohort.  A larger value means that a larger share of the
    detected beats has morphology inconsistent with the record's median beat.
    NaN is returned when too few complete, finite beats are available.
    """
    if (
        not np.isfinite(correlation_threshold)
        or correlation_threshold < -1.0
        or correlation_threshold > 1.0
    ):
        raise ValueError("correlation_threshold must be finite and between -1 and 1")

    correlations = _beat_template_correlations(
        ecg,
        r_peak_samples,
        sampling_rate_hz,
        before_peak_s=before_peak_s,
        after_peak_s=after_peak_s,
        min_beats=min_beats,
    )
    if len(correlations) < min_beats:
        return float("nan")
    return float(np.mean(correlations < correlation_threshold))


def low_template_correlation_fraction_leave_one_out(
    ecg: np.ndarray | list[float],
    r_peak_samples: np.ndarray | list[float],
    sampling_rate_hz: float,
    correlation_threshold: float = 0.50,
    before_peak_s: float = 0.20,
    after_peak_s: float = 0.40,
    min_beats: int = 3,
) -> float:
    """Return the low-correlation beat fraction using leave-one-out templates."""
    if (
        not np.isfinite(correlation_threshold)
        or correlation_threshold < -1.0
        or correlation_threshold > 1.0
    ):
        raise ValueError("correlation_threshold must be finite and between -1 and 1")

    correlations = _beat_template_correlations(
        ecg,
        r_peak_samples,
        sampling_rate_hz,
        before_peak_s=before_peak_s,
        after_peak_s=after_peak_s,
        min_beats=min_beats,
        leave_one_out=True,
    )
    if len(correlations) < min_beats:
        return float("nan")
    return float(np.mean(correlations < correlation_threshold))


def low_template_correlation_severity_leave_one_out(
    ecg: np.ndarray | list[float],
    r_peak_samples: np.ndarray | list[float],
    sampling_rate_hz: float,
    correlation_threshold: float = 0.50,
    before_peak_s: float = 0.20,
    after_peak_s: float = 0.40,
    min_beats: int = 3,
) -> float:
    """Return the mean normalized correlation deficit below a fixed threshold.

    A beat at the threshold contributes zero, while a perfectly anticorrelated
    beat contributes one.  Beats above the threshold contribute zero.  This
    retains how severe each poor match is instead of counting every poor match
    equally.
    """
    if (
        not np.isfinite(correlation_threshold)
        or correlation_threshold <= -1.0
        or correlation_threshold > 1.0
    ):
        raise ValueError("correlation_threshold must be finite and in (-1, 1]")

    correlations = _beat_template_correlations(
        ecg,
        r_peak_samples,
        sampling_rate_hz,
        before_peak_s=before_peak_s,
        after_peak_s=after_peak_s,
        min_beats=min_beats,
        leave_one_out=True,
    )
    if len(correlations) < min_beats:
        return float("nan")
    deficits = np.clip(
        (correlation_threshold - correlations) / (correlation_threshold + 1.0),
        0.0,
        1.0,
    )
    return float(np.mean(deficits))


def demo() -> None:
    """Run a small sanity check without needing real ECG data."""
    regular_rr = np.array([0.80, 0.81, 0.79, 0.80, 0.81])
    irregular_rr = np.array([0.55, 1.10, 0.68, 1.25, 0.72])

    regular_value = normalized_rr_mad(regular_rr)
    irregular_value = normalized_rr_mad(irregular_rr)

    print(f"Regular RR normalized MAD:   {regular_value:.4f}")
    print(f"Irregular RR normalized MAD: {irregular_value:.4f}")

    # This checks only that the implementation behaves sensibly on toy data.
    # It does not demonstrate classification performance on the real dataset.
    assert irregular_value > regular_value
    print("Sanity check passed: the irregular example has the larger value.")


if __name__ == "__main__":
    demo()
