"""
tracks.ecg_cinc2017 — second reference track: single-lead ECG rhythm (AF).

Real data: PhysioNet/CinC Challenge 2017 — ~8,528 short single-lead ECGs (AliveCor,
300 Hz), labelled Normal (N) / AF (A) / Other rhythm (O) / Noisy (~). Open, direct
download (no agreement). `load()` uses `wfdb` (Colab); `smoke()` synthesises the four
rhythm classes so CI/offline runs green.

DSP the student does: band-pass + 50/60 Hz notch, Pan–Tompkins QRS detection, R–R
irregularity / HRV, and signal-quality features (the AF signature is an *irregularly
irregular* rhythm; Noisy is low signal quality).
"""
from __future__ import annotations
import numpy as np

from adapter import (TrackAdapter, TrackMeta, Recording, default_baseline,
                     bandpass_notch, wavelet_denoise, denoise, spectral_bandpower,
                     DENOISE_CFG_KEYS, SPECTRAL_CFG_KEYS)
import biosignals as bio


def _gauss(t, a, mu, s):
    return a * np.exp(-0.5 * ((t - mu) / s) ** 2)


def _beats(rr_list, fs, dur, rng, p_amp=0.10, fib=0.0):
    n = int(dur * fs); t = np.arange(n) / fs; x = np.zeros(n)
    tc = 0.4
    for rr in rr_list:
        if tc > dur - 0.3:
            break
        x += _gauss(t, 1.0, tc, 0.010) - _gauss(t, 0.22, tc + 0.035, 0.012)      # QRS
        if p_amp > 0:
            x += _gauss(t, p_amp, tc - 0.18, 0.025)                              # P
        x += _gauss(t, 0.28, tc + 0.16, 0.040)                                    # T
        tc += rr
    if fib:                                                                       # fibrillatory baseline (AF)
        x += fib * np.sin(2 * np.pi * rng.uniform(5, 8) * t) * rng.standard_normal(n) * 0.3
    x += 0.05 * np.sin(2 * np.pi * 0.3 * t)                                        # mild baseline wander
    return x


def _synth_record(kind, fs, dur, rng, typicality=1.0):
    """One synthetic record of class `kind`.

    `typicality` in [0, 1] is how *textbook* this example is: 1.0 is the clean
    caricature (unmistakable AF, unmistakable noise), 0.0 is the borderline case a
    real cardiologist argues about. Drawing it uniformly per record is what turns
    a trivially separable smoke set into an honest one — classes now OVERLAP, and
    the N↔A confusion that dominates the real challenge shows up here too.
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    a = float(np.clip(typicality, 0.0, 1.0))
    rr0 = rng.uniform(0.60, 1.05)                    # this subject's resting R-R
    nbeat = int(dur / rr0) + 20
    # every real recording carries some noise, hum and wander — not only the '~' class,
    # otherwise "has 50 Hz hum" is a giveaway label rather than a signal-quality feature.
    base_noise = rng.uniform(0.03, 0.22)
    hum = rng.uniform(0.0, 0.25)
    wander = rng.uniform(0.02, 0.25)

    if kind == "N":                              # normal sinus rhythm + mild HRV
        hrv = rng.uniform(0.015, 0.080)
        rr = rr0 + 0.05 * np.sin(2 * np.pi * 0.25 * np.arange(nbeat)) + hrv * rng.standard_normal(nbeat)
        if rng.random() < 0.4:                   # ectopic beats on an otherwise normal strip
            j = rng.integers(0, nbeat - 1, size=int(rng.integers(1, 5)))
            rr[j] *= 0.55
            rr[np.clip(j + 1, 0, nbeat - 1)] *= 1.4
        x = _beats(rr, fs, dur, rng, p_amp=0.04 + 0.10 * a)
    elif kind == "A":                            # AF: irregularly irregular, no P waves
        scale = 0.05 + 0.17 * a                  # low typicality -> AF that looks almost regular
        rr = np.clip(rr0 * 0.75 + rng.exponential(scale, nbeat), 0.28, 1.7)
        x = _beats(rr, fs, dur, rng, p_amp=0.05 * (1 - a), fib=0.03 + 0.10 * a)
    elif kind == "O":                            # Other: bigeminy, or regular brady/tachycardia
        if rng.random() < 0.5:
            depth = 0.06 + 0.42 * a              # shallow bigeminy is easily read as normal HRV
            rr = (np.tile([rr0 * (1 - depth), rr0 * (1 + depth)], nbeat // 2 + 1)[:nbeat]
                  + 0.03 * rng.standard_normal(nbeat))
        else:
            rr = (np.full(nbeat, rr0 * (0.62 if rng.random() < 0.5 else 1.45))
                  + 0.03 * rng.standard_normal(nbeat))
        x = _beats(rr, fs, dur, rng, p_amp=0.10)
    else:                                        # ~ Noisy: normal rhythm + corruption (low SQI)
        rr = rr0 + 0.04 * rng.standard_normal(nbeat)
        x = _beats(rr, fs, dur, rng, p_amp=0.10)
        nz = 0.10 + 0.55 * a                     # a barely-noisy '~' is a genuinely hard call
        base_noise = max(base_noise, nz)
        if rng.random() < 0.6:                   # some '~' are motion artifact only, no mains hum
            hum = max(hum, 0.3 * nz)
        wander = max(wander, 0.6 * nz)

    return (x + base_noise * rng.standard_normal(n)
            + hum * np.sin(2 * np.pi * 50 * t)
            + wander * np.sin(2 * np.pi * 0.4 * t))


def _rpeaks(ecg, fs):
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], "band")
    f = filtfilt(b, a, ecg)
    d = np.diff(f, prepend=f[0]) ** 2
    w = max(1, int(0.15 * fs))
    integ = np.convolve(d, np.ones(w) / w, "same")
    pk, _ = find_peaks(integ, height=0.3 * np.max(integ) + 1e-9, distance=int(0.3 * fs))
    return pk

def _rpeaks_adaptive(ecg, fs):
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], "band")
    f = filtfilt(b, a, ecg)
    d = np.diff(f, prepend=f[0]) ** 2
    w = max(1, int(0.15 * fs))
    integ = np.convolve(d, np.ones(w) / w, "same")
    candidates, _ = find_peaks(integ, height=0.3 * np.max(integ) + 1e-9, distance=int(0.3 * fs))

    # Step 5: Candidate detection
    candidates, _ = find_peaks(integ, distance = int(0.2 * fs))

    # Step 6: Adaptive threshold
    if len(candidates) == 0:
        return np.array([], dtype = int)

    initial_candidates = candidates[candidates < int(fs * 5)]
    if len(initial_candidates) < 3:
        initial_candidates = candidates[:10]

    initial_values = integ[initial_candidates]
    NPKI, SPKI = np.percentile(initial_values, (20, 90))
    T = NPKI + 0.25 * (SPKI - NPKI)

    rpeaks = []
    refractory = 0.3 * fs
    alpha = 0.125

    for candidate in candidates:
        amplitude = integ[candidate]
        T = NPKI + 0.25 * (SPKI - NPKI)

        if amplitude > T:
            if not rpeaks or candidate - rpeaks[-1] >= refractory:
                rpeaks.append(candidate)
                SPKI = alpha * amplitude + (1 - alpha) * SPKI
            else:
                NPKI = alpha * amplitude + (1 - alpha) * NPKI

        else:
            NPKI = alpha * amplitude + (1 - alpha) * NPKI

    return np.asarray(rpeaks, dtype=int)


def _rpeaks_improved(ecg, fs):
    """
    Idea:
        Original ECG
        |
        v
    Bandpass filtering
        |
        v
    Derivative + squaring + integration
        |
        v
    Adaptive QRS detection
        |
        v
    Approximate QRS positions
        |
        v
    Search near each QRS position
        |
        v
    Refined R-peak positions
    """
    from scipy.signal import butter, sosfiltfilt, find_peaks

    # Step 1: Bandpass filtering
    sos = butter(3, [5 / (fs /2), 15 / (fs / 2)], "band", output = "sos")
    filtered_ecg = sosfiltfilt(sos = sos, x = ecg)

    # Step 2: Differentiation
    diffed_ecg = np.diff(filtered_ecg, prepend=filtered_ecg[0])

    # Step 3: Squaring
    sqrd_ecg = diffed_ecg ** 2

    # Step 4: Moving-window integration
    w = max(1, int(0.15 * fs))
    integrated_ecg = np.convolve(sqrd_ecg, np.ones(w) / w , "same")

    # Step 5: Candidate detection
    candidates, _ = find_peaks(integrated_ecg, distance = int(0.2 * fs))

    # Step 6: Adaptive threshold
    if len(candidates) == 0:
        return np.array([], dtype = int)

    initial_candidates = candidates[candidates < int(fs * 5)]
    if len(initial_candidates) < 3:
        initial_candidates = candidates[:10]

    initial_values = integrated_ecg[initial_candidates]
    NPKI, SPKI = np.percentile(initial_values, (20, 90))
    T = NPKI + 0.25 * (SPKI - NPKI)

    rpeaks = []
    refractory = 0.3 * fs
    alpha = 0.125

    for candidate in candidates:
        amplitude = integrated_ecg[candidate]
        T = NPKI + 0.25 * (SPKI - NPKI)

        if amplitude > T:
            if not rpeaks or candidate - rpeaks[-1] >= refractory:
                rpeaks.append(candidate)
                SPKI = alpha * amplitude + (1 - alpha) * SPKI
            else:
                NPKI = alpha * amplitude + (1 - alpha) * NPKI

        else:
            NPKI = alpha * amplitude + (1 - alpha) * NPKI
    
    # Step 7: R-peak localization

    refined_peaks = []
    radius = int(0.1 * fs)

    for rpeak in rpeaks:
        start = max(0, rpeak - radius)
        end = min(len(filtered_ecg), rpeak + radius + 1)

        local_peak = np.argmax(np.abs(filtered_ecg[start : end]))
        global_peak = local_peak + start

        if global_peak not in refined_peaks:
            refined_peaks.append(global_peak)

    # Step 8: Missed-beat detection
    suspicious_gaps = np.array([], dtype=int)
    recovered_peaks = []
    if len(refined_peaks) >= 6:
        rr = np.diff(refined_peaks) / fs
        rr_threshold = 1.8 * np.median(rr)

        suspicious_gaps = np.where(rr > rr_threshold)[0]
        
        for gap in suspicious_gaps:

            gap_start = refined_peaks[gap]
            gap_end = refined_peaks[gap +1]
            mask = (candidates > gap_start + refractory) & (candidates < gap_end - refractory)
            gap_candidates = candidates[mask]

            if gap_candidates.size == 0: 
                continue

            gap_amplitudes = integrated_ecg[gap_candidates]
            best_idx = np.argmax(gap_amplitudes)

            best_candidate = gap_candidates[best_idx]
            best_amplitude = gap_amplitudes[best_idx]

            search_threshold = NPKI + 0.125 * (SPKI - NPKI)
            if best_amplitude > search_threshold:

                start = max(0, best_candidate - radius)
                end = min(len(filtered_ecg), best_candidate + radius + 1)

                local_peak = np.argmax(np.abs(filtered_ecg[start : end]))
                global_peak = local_peak + start

                if global_peak not in refined_peaks and global_peak not in recovered_peaks and global_peak - gap_start >= refractory and gap_end - global_peak >= refractory:
                    recovered_peaks.append(global_peak)

    # Step 9: Return detected R-peaks
    rpeaks_combined = np.concatenate((refined_peaks, recovered_peaks), axis=0)
    rpeaks_int = np.asarray(rpeaks_combined, dtype=int)
    rpeaks_unique = np.unique(rpeaks_int)

    return rpeaks_unique


def _try_new_rpeaks(ecg,fs):
    rpeaks_found = []

    # width QRS-complex 70-110 ms -> assume wide at 120 ms = 0.12 s
    qrs_wid = int(0.12 * fs/2) * 2 # make sure it's dividible by 2          ca. 36-37 frames

    ecg_defined = 20 * np.gradient(ecg)
    ecg_biss = np.gradient(ecg_defined)

    perc95 = np.percentile(ecg,97)
    perc05 = np.percentile(ecg,3)

    for i in range(len(ecg_defined)):
        # check if under 05
        if ecg_defined[i] < perc05:
            # check if over 95 within qrs_wid
    
            for j in range(max(int(i-qrs_wid/2),0),min(int(i+qrs_wid/2),len(ecg_defined))):
                if ecg_defined[j] > perc95:
                    # have heart beat between indexes i and j
                    # find flat point
            
                    the_min_val = abs(ecg_biss[i])
                    the_min_k = i
                    for k in range(i,j):
                        if abs(ecg_biss[k]) > the_min_val:
                            the_min_k = k
                    rpeaks_found.append(the_min_k)

    rpeaks_found = np.unique(rpeaks_found)
    if len(rpeaks_found) == 0:
        return np.array([], dtype = int)

    # define first peaks, and last peaks
    the_peak_width = 80

    IO_rpeaks_first = np.zeros(len(rpeaks_found))
    IO_rpeaks_first[0] = 1
    IO_rpeaks_last = np.zeros(len(rpeaks_found))
    IO_rpeaks_last[-1] = 1
    for i in range(len(rpeaks_found)-1):
        if rpeaks_found[i+1] > rpeaks_found[i] + the_peak_width:
            IO_rpeaks_first[i+1] = 1
            IO_rpeaks_last[i] = 1


    # combine to see where the first ones are put
    rpeaks_first = np.multiply(IO_rpeaks_first, rpeaks_found)
    rpeaks_last = np.multiply(IO_rpeaks_last, rpeaks_found)

    rpeaks_first = rpeaks_first[rpeaks_first != 0]
    rpeaks_last = rpeaks_last[rpeaks_last != 0]

    rpeaks_first = rpeaks_first.astype(int)
    rpeaks_last = rpeaks_last.astype(int)
    rpeaks_final = []

    for i in range(len(rpeaks_first)):
        if rpeaks_first[i] == rpeaks_last[i]:
            rpeaks_final.append(int(rpeaks_first[i]))
        else:
            peak_div = ecg_defined[int(rpeaks_first[i]):int(rpeaks_last[i])]
            peak_div = abs(peak_div - np.mean(peak_div))
            extrm_div = np.argmax(peak_div)
            rpeaks_final.append(int(extrm_div))


    r_returns_uwu = []
    for i in range(len(rpeaks_first)):
        # find max in there
        qrs_peak = ecg_defined[rpeaks_first[i]:min((rpeaks_last[i]+1),(len(rpeaks_last)-1))]
        if len(qrs_peak) < 3:
            r_returns_uwu.append(rpeaks_first[i])       # if less than 3 detected frames on peak -> take first frame as peak
        else : 
            qrs_peak = abs(qrs_peak - np.mean(qrs_peak))
            qrs_max_index = np.argmax(qrs_peak)
            qrs_max_index_tot = qrs_max_index + rpeaks_first[i]
            r_returns_uwu.append(qrs_max_index_tot)
    return np.array(r_returns_uwu)


def _divisible_len(ecg,fs):
    # XXXXXXX THIS ONE WORKS PRETTY WELL !!!!!!!!!!!!!
    divisions_nr = 5
    list_o_ecg = np.array_split(ecg,divisions_nr)

    IO_index = np.array([])
    for i in range(len(list_o_ecg)):
        r_peaks_o = _try_new_rpeaks(list_o_ecg[i],fs)

        # have r-peaks w. wrong numbers
        IO_segment = np.zeros(len(list_o_ecg[i]))
        for j in r_peaks_o:
            if (j - int(j)) == 0.0:

                IO_segment[int(j)] = 1
        IO_index = np.concatenate((IO_index,IO_segment))


    r_peaks_now = []
    for i in range(len(IO_index)):
        if IO_index[i] == 1:
            r_peaks_now.append(i)

    return np.array(r_peaks_now)

def _MAD(rpeaks):
    # 
    RRI = []
    for i in range(len(rpeaks)-1):
        RRI[i] = abs(rpeaks[i]-rpeaks[i+1])

    RRI = np.array(RRI) - np.median(RRI)
    return(np.median(RRI))


def _rr_sample_entropy(rr, m=2, r_ratio=0.2):

    rr = np.asarray(rr, dtype=float)

    if len(rr) < 20 or not np.all(np.isfinite(rr)) or np.any(rr <= 0):
        return np.nan
    
    r = r_ratio *np.std(rr)

    rr_seq_m, rr_seq_m1 = [], []
    A, B = 0, 0
    for i in range(len(rr) - m):
        rr_seq_m.append(rr[i : i+m])

    for i in range(len(rr) - m):
        rr_seq_m1.append(rr[i : i + m + 1])
    
    for i in range(len(rr_seq_m)-1):
        for j in range(i + 1, len(rr_seq_m)):
            pair1 = rr_seq_m[i]
            pair2 = rr_seq_m[j]
            if np.max(np.abs(pair1-pair2)) <= r:
                B += 1

    for i in range(len(rr_seq_m1)-1):
            for j in range(i + 1, len(rr_seq_m1)):
                pair1 = rr_seq_m1[i]
                pair2 = rr_seq_m1[j]
                if np.max(np.abs(pair1-pair2)) <= r:
                    A += 1

    if A == 0 or B == 0:
        return np.nan

    sampen = -np.log(A / B)

    return sampen

def _rr_difference_entropy(rr):
    rr = np.asarray(rr, dtype=float)
    if len(rr) < 10 or not np.all(np.isfinite(rr)) or np.any(rr <= 0):
        return np.nan

    drr_normalized = np.diff(rr)/np.median(rr)

    bins = np.array([
        -np.inf,
        -0.4,
        -0.2,
        -0.1,
        0.1,
        0.2,
        0.4,
        np.inf
    ])

    counts, edges = np.histogram(drr_normalized, bins=bins)

    p = counts / np.sum(counts)
    p_nonzero = p[p> 0]
    #shannon entropy
    H = -np.sum(p_nonzero * np.log(p_nonzero))
    H_normalized = H / np.log(len(counts))
    return H_normalized

def _qrs_morphology_correlations(ecg, peaks, fs, min_beats=5):
    ecg = np.asarray(ecg, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    if fs <= 0:
        return np.nan

    before = int(round(0.12 * fs))
    after = int(round(0.16 * fs))

    segments = []

    for peak in peaks:
        start = peak - before
        end = peak + after
        if start < 0 or end > len(ecg):
            continue
        segment = ecg[start : end]
        if np.all(np.isfinite(segment)):
            segments.append(segment)

    if len(segments) < min_beats:
        return np.nan

    matrix = np.asarray(segments, dtype=float)
    template = np.median(matrix, axis=0)
    template_centered = template - np.mean(template)

    magnitude = np.linalg.norm(template_centered)
    if magnitude == 0:
        return np.nan

    correlations = []
    for segment in matrix:
        centered = segment - np.mean(segment)
        denominator = np.linalg.norm(centered) * magnitude
        if denominator == 0:
            continue

        p = np.dot(centered, template_centered) /denominator
        correlations.append(p)

    if len(correlations) < min_beats:
        return np.nan

    return correlations


def _low_template_correlation_fraction(
    ecg, peaks, fs, threshold=0.50, before_s=0.20, after_s=0.40, min_beats=3
):
    """Fraction of beats whose correlation with the median beat is below threshold."""
    before, after = int(round(before_s * fs)), int(round(after_s * fs))
    beats = [
        np.asarray(ecg[peak - before : peak + after], dtype=float)
        for peak in np.asarray(peaks, dtype=int)
        if peak - before >= 0 and peak + after <= len(ecg)
    ]
    beats = [beat for beat in beats if np.all(np.isfinite(beat))]
    if len(beats) < min_beats:
        return np.nan

    template = np.median(np.asarray(beats), axis=0)
    correlations = np.asarray(
        [np.corrcoef(beat, template)[0, 1] for beat in beats], dtype=float
    )
    correlations = correlations[np.isfinite(correlations)]
    return float(np.mean(correlations < threshold)) if correlations.size >= min_beats else np.nan


def _short_long_rr_pair_ratio(
    rr, window=12, short_ratio=0.80, long_ratio=1.20, min_intervals=6
):
    """Fraction of adjacent RR pairs that form a short-then-long pattern."""
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < min_intervals:
        return np.nan

    matches = []
    half = window // 2
    for index in range(rr.size - 1):
        start = max(0, index - half)
        stop = min(rr.size, index + 2 + half)
        neighbours = np.concatenate((rr[start:index], rr[index + 2 : stop]))
        if neighbours.size < 3:
            continue
        reference = float(np.median(neighbours))
        if reference > np.finfo(float).eps:
            matches.append(
                rr[index] < short_ratio * reference
                and rr[index + 1] > long_ratio * reference
            )

    return float(np.mean(matches)) if matches else np.nan


def main():

    rr_regular = np.array([0.8] * 30)

    rr_alternating = np.array([0.6, 1.0] * 15)

    rng = np.random.default_rng(0)
    rr_random = rng.uniform(0.5, 1.2, 30)

    print("Regular:", _rr_difference_entropy(rr_regular))
    print("Alternating:", _rr_difference_entropy(rr_alternating))
    print("Random:", _rr_difference_entropy(rr_random))

if __name__ == "__main__":
    main()


#: Hold-out records: 30 per class (N/A/O/~), drawn with a fixed seed from records that none of the
#: default downloads fetch. Never train, tune or select on them; predict them once at the end.
#: See HOLDOUT_EVALUATION.md.
HOLDOUT_RECORDS = (
    "A00015", "A00236", "A00308", "A00370", "A00445", "A00482", "A00592", "A00711", "A00722", "A00766",
    "A00774", "A00858", "A01020", "A01063", "A01116", "A01405", "A01406", "A01461", "A01499", "A01507",
    "A01578", "A01591", "A01607", "A01805", "A01941", "A02119", "A02212", "A02346", "A02375", "A02400",
    "A02470", "A02475", "A02676", "A02771", "A02841", "A02924", "A02926", "A02945", "A03003", "A03004",
    "A03043", "A03081", "A03103", "A03128", "A03162", "A03319", "A03403", "A03443", "A03580", "A03596",
    "A03756", "A03792", "A03860", "A03934", "A03976", "A04186", "A04220", "A04276", "A04427", "A04576",
    "A04652", "A04685", "A04771", "A04816", "A04899", "A04941", "A04984", "A05017", "A05263", "A05267",
    "A05299", "A05331", "A05387", "A05507", "A05816", "A05960", "A05977", "A05989", "A06002", "A06053",
    "A06156", "A06217", "A06259", "A06261", "A06262", "A06297", "A06304", "A06537", "A06583", "A06614",
    "A06774", "A06776", "A06804", "A06830", "A06841", "A06918", "A06958", "A07110", "A07139", "A07193",
    "A07198", "A07256", "A07388", "A07595", "A07664", "A07708", "A07744", "A07763", "A07786", "A07859",
    "A07901", "A07903", "A07904", "A07983", "A08043", "A08176", "A08195", "A08228", "A08361", "A08493",
)

class ECGCinC2017Track(TrackAdapter):
    meta = TrackMeta(
        name="ECG rhythm (AF)",
        dataset="PhysioNet/CinC Challenge 2017 (single-lead ECG)",
        dataset_version="1.0.0",
        license="Open (PhysioNet Challenge)",
        citation="Clifford GD, et al. PhysioNet/CinC Challenge 2017.",
        url="https://physionet.org/content/challenge-2017/1.0.0/",
        signals=["single-lead ECG @ 300 Hz"],
        task_type="multiclass",
        classes=["N", "A", "O", "~"],           # Normal, AF, Other, Noisy
        split_unit="record",
        default_metrics=["macro_f1", "cohens_kappa"],
        smoke_test_records=["A00001", "A00004", "A00006"],
        holdout_ids=HOLDOUT_RECORDS,                            # see HOLDOUT_EVALUATION.md
        expected_runtime="~2–5 min on a subset (Colab, CPU)",
        dsp_focus="band-pass + notch, Pan–Tompkins QRS, R–R irregularity/HRV, signal quality",
        difficulty=3,
        eval_modes=("new-record",),
        submission_granularity="record",         # one label per record for hold-out evaluation
    )

    #: Capability declaration (see `TrackAdapter.SUPPORTED_CFG_KEYS`). Like
    #: Sleep-EDF, this track's `preprocess()` forwards the whole Chapter 8
    #: denoise menu, and its HRV/quality features include band powers integrated
    #: from a Chapter 7 estimator — so both stage-2 and stage-3 menus are real.
    SUPPORTED_CFG_KEYS = DENOISE_CFG_KEYS | SPECTRAL_CFG_KEYS | {"ecg_band", "r_peak", "include_short_long_rr_pair", "include_low_template_correlation", "classifier", "n_estimators", "n_neighbors", "weights", "C", "kernel", "min_samples_leaf", "max_depth", "min_samples_leaf", "gamma"}

    #: Class counts for the synthetic smoke set, in the same lopsided shape as the
    #: real challenge (Normal dominates; AF and Noisy are rare). The rubric grades
    #: how you address that imbalance — it cannot be graded on a balanced fake.
    SMOKE_COUNTS = {"N": 34, "A": 16, "O": 24, "~": 12}

    def smoke(self, per_class=None, fs=300, dur=20.0):
        """Synthetic four-class smoke set with a per-record **typicality** draw, so
        the classes overlap the way they do on the real challenge instead of being
        four separable caricatures — and with the real challenge's **class
        imbalance** (see `SMOKE_COUNTS`). GroupKFold-by-record lands around
        macro-F1 ≈ 0.7 (below the 0.83 challenge-winner yardstick, as it should
        be), with N and A the pair that trades places — exactly the confusion the
        notebook asks you to predict before running.

        Pass `per_class=<int>` for the old balanced cohort if you specifically
        want to isolate a change from the imbalance effect."""
        recs, k = [], 0
        for cls in self.meta.classes:
            n_cls = per_class if per_class is not None else self.SMOKE_COUNTS.get(cls, 20)
            for _ in range(n_cls):
                r = np.random.default_rng(1000 + k)
                x = _synth_record(cls, fs, dur, r, typicality=r.uniform(0.0, 1.0))
                recs.append(Recording(group=f"R{k:04d}", fs=fs,
                                      epochs={"ecg": x[None, :]}, labels=np.array([cls])))
                k += 1
        return recs

    # ---- module 2: PREPROCESSING (opt-in; the recipe is your decision) ----
    def preprocess(self, rec: Recording, cfg=None) -> Recording:
        """Identity by default — the shipped baseline lets `_rpeaks()` own its own
        band-pass and lets every other feature see the raw strip. That is a
        starting point, not a recommendation, and hoisting the cleaning up here is
        one of the decisions this track exists to make you argue for.

        **Two supplied recipes, answering different noise signatures.** Pick with
        `cfg["preprocess"]`:

            ECGCinC2017Track(cfg={"preprocess": "none"})                             # default
            ECGCinC2017Track(cfg={"preprocess": "bandpass", "ecg_band": (0.5, 40.0), "notch": 50.0})
            ECGCinC2017Track(cfg={"preprocess": "wavelet",  "wavelet": "db4"})
            ECGCinC2017Track(cfg={"preprocess": "denoise",  "impulsive": "median",
                                  "baseline": "highpass", "hp_fc": 0.5,
                                  "powerline": "notch"})

        | `preprocess` | attacks | on THIS track that means | costs you |
        |---|---|---|---|
        | `"bandpass"` | **stationary, narrow-band** interference | 50 Hz mains hum and slow baseline wander are exactly that: fixed in frequency, present throughout. A 0.5-40 Hz band plus a notch removes both for almost nothing | the **QRS complex is a broadband, ~80 ms transient** — band-limiting rounds its edges, widens its apparent duration and shifts the peak the R-R interval is measured from. And your powerline-power feature was *detecting* the hum: notch it away and you have deleted the evidence the `~` class rests on |
        | `"wavelet"` | **non-stationary transients** | motion artifact, electrode contact loss, muscle bursts — the corruption that *defines* the Noisy (`~`) class. These are broadband and brief, so no band contains them; DWT thresholding removes them where they are, in time and scale | over-thresholding erodes low-amplitude morphology (the P wave whose *absence* is the AF signature) and, again, quietly cleans away the very corruption the `~` class is labelled for. Cleaning the noisy class into looking normal is a way to lose the problem, not solve it |
        | `"none"` *(default)* | nothing | detection owns its filter; features see the raw signal | the filter is invisible in the report and cannot be swapped as a stage |

        **`"denoise"` is the per-noise-type menu** (Chapter 8's own structure: one
        key per corruption, not one key per technique), and ECG is the track it
        fits best, because a real single-lead strip carries *all four* of Ch. 8's
        categories at once and they want different remedies:

        | what you see (§8.3's three lenses) | Ch. 8 category | key |
        |---|---|---|
        | brief huge outliers, broadband transient | electrode pop / impulsive (§8.7) | `impulsive="median"` — and it runs FIRST, because §9.7 says a linear filter cannot delete a spike, it only smears it |
        | slow drift under the trace | baseline wander (§8.8) | `baseline="highpass"` at 0.5 Hz for rhythm work — but §8.8 warns a diagnostic ECG needs content down to 0.05 Hz, and too aggressive a corner "can manufacture artificial ST shifts that mimic ischaemia" |
        | fine steady buzz, line at 50/60 Hz | powerline (§8.6) | `powerline="notch"` — "the friendliest artifact" |
        | structureless fuzz across the band | white/broadband (§8.4) | `broadband="wavelet"` to keep the QRS sharp, or `"savgol"` to keep peak height with a linear filter |

        The sharpest lesson on this track is that **denoising can destroy the
        label**. Two of the eleven shipped features (the QRS/high-frequency ratio
        and the 49-51 Hz powerline power) exist to *measure* corruption. Any
        preprocessing that removes corruption before those features are computed
        makes the `~` class harder, not easier. Measure it both ways and report
        which you chose.

        Note: `bandpass_notch` is zero-phase (`filtfilt`) and therefore
        **non-causal** — legitimate here, because CinC-2017 is retrospective
        analysis of complete 30 s strips. A wearable rhythm monitor deciding in
        real time could not use it; see `adapter.bandpass_notch.__doc__` and pass
        `cfg={"causal": True}` to measure what a deployable filter costs.
        """
        cfg = self._cfg(cfg)

        #Enables usage of multiple preprocessing filters
        pre = cfg.get("preprocess")
        if isinstance(pre, list):
            for i in pre:
                rec = self.preprocess(rec, cfg = i)
            return rec
      
        how = str(cfg.get("preprocess", "none") or "none").lower()
        if how in ("denoise", "by_noise", "noise"):
            return denoise(rec, impulsive=cfg.get("impulsive"), baseline=cfg.get("baseline"),
                           powerline=cfg.get("powerline"), broadband=cfg.get("broadband"),
                           note=self.note,
                           **{k: cfg[k] for k in ("hp_fc", "f0", "q", "win", "wavelet", "causal",
                                                  "poly_order", "savgol_order", "harmonics", "mu",
                                                  "order", "threshold_mode", "level")
                              if cfg.get(k) is not None})
        if how in ("wavelet", "dwt", "wavelet_denoise"):
            return wavelet_denoise(rec, wavelet=cfg.get("wavelet", "db4"),
                                   level=cfg.get("wavelet_level"),
                                   threshold_mode=cfg.get("wavelet_mode", "soft"))
        band, notch = cfg.get("ecg_band"), cfg.get("notch")
        if how in ("bandpass", "bandpass_notch", "filter") and band is None and notch is None:
            band = (0.5, 40.0)
            self.note("preprocess='bandpass' with no ecg_band given — using 0.5-40 Hz. Set "
                      "cfg['ecg_band'] yourself and justify the edges; note that this band also "
                      "removes the 50 Hz content one of your features measures.")
        if band is None and notch is None:
            return rec
        return bandpass_notch(rec, band=band, notch=notch,
                              causal=bool(cfg.get("causal", False)))

    def extract_features(self, rec: Recording, cfg=None):
        """Module 3 — one feature row per RECORD (R-R/HRV descriptors + a
        signal-quality index). Where the band-pass that feeds R-peak detection
        lives is a design choice: inside `_rpeaks` (as here, so detection owns its
        own filter) or hoisted into `preprocess()` so every feature sees the same
        cleaned signal — `cfg={"prepro
        
        cess": "bandpass"}` now does exactly that.
        Both are defensible; the second makes the filter a reportable, swappable
        stage.

        **Three of the eleven features are band powers, so the Ch. 7 spectral
        estimator is a knob here too** — `cfg["spectral_method"]` ∈
        `{"periodogram", "bartlett", "welch"(default), "multitaper", "ar"}`; see
        `adapter.make_spectral_estimator`. The trade-off has a specific shape on
        this track. The 49-51 Hz powerline feature is the narrowest band in the
        whole scaffold — 2 Hz wide — so it is the one most punished by an
        estimator whose resolution has coarsened, and the one most helped by
        `"multitaper"` on a short strip. Conversely `"ar"` will happily place a
        pole near 50 Hz and report a confident, smooth, possibly fictitious hum
        peak, which is §7.9's over-fitting warning arriving as a feature value.
        A 20-30 s strip at 300 Hz is long enough for Welch to be comfortable, so
        expect small movements — say so honestly if that is what you measure."""
        from scipy.stats import kurtosis
        cfg = self._cfg(cfg)

        method = str(cfg.get("spectral_method", "welch") or "welch").lower()
        mkw = {"ar_order": int(cfg.get("ar_order", 16)),
               "bandwidth": float(cfg.get("mt_bandwidth", 4.0))}
        rpeak_method = str(cfg.get("r_peak"))

        def _bp(sig, f_s, band):
            if method in ("welch", "default"):
                return bio.bandpower(sig, f_s, band)
            return spectral_bandpower(sig, f_s, band, method=method, **mkw)


        ecg = rec.epochs["ecg"][0]
        fs = rec.fs

        if rpeak_method in ("adaptive", "R1"):
            pk = _rpeaks_adaptive(ecg, fs)
        elif rpeak_method in ("advanced", "R2"):
            pk = _rpeaks_improved(ecg, fs)
        elif rpeak_method in ("R3"):
            pk = _divisible_len(ecg, fs)
        else:
            pk = _rpeaks(ecg, fs)

        rr = np.diff(pk) / fs
        qrs_morph = _qrs_morphology_correlations(ecg, pk, fs)
        include_short_long_rr_pair = bool(
            cfg.get("include_short_long_rr_pair", False)
        )
        if include_short_long_rr_pair:
            short_long_rr_pair = _short_long_rr_pair_ratio(rr)
        include_low_template_correlation = bool(
            cfg.get("include_low_template_correlation", False)
        )
        if include_low_template_correlation:
            low_template_correlation = _low_template_correlation_fraction(ecg, pk, fs)

        if len(rr) >= 3:
            hr = 60.0 / np.mean(rr); sdnn = np.std(rr) * 1000
            rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000
            pnn50 = np.mean(np.abs(np.diff(rr)) > 0.05) * 100
            cv = np.std(rr) / (np.mean(rr) + 1e-9)
            irr = np.mean((rr < 0.6) | (rr > 1.2))
            #Added features
            sampen = _rr_sample_entropy(rr)
            drren = _rr_difference_entropy(rr)
            qrsstd =  float(np.std(qrs_morph))
            qrsmed = float(np.median(qrs_morph))

        else:
            hr = sdnn = rmssd = pnn50 = cv = irr = 0.0
            #Added features
            sampen = 0.0
            drren = 0.0
            qrsstd = 0.0
            qrsmed = 0.0

        qrs_bp = _bp(ecg, fs, (5, 15))
        hf = _bp(ecg, fs, (40, min(fs / 2 - 1, 90)))
        feats = [hr, sdnn, rmssd, pnn50, cv, irr, float(len(rr)),
                 qrs_bp / (hf + 1e-9),                      # signal-quality index
                 _bp(ecg, fs, (49, 51)),                    # powerline power
                 float(np.std(ecg)), float(kurtosis(ecg)), 
                 #Added features
                 sampen,
                 drren,
                 #qrsstd,
                 qrsmed,
                 ]
        if include_short_long_rr_pair:
            feats.append(short_long_rr_pair)
        if include_low_template_correlation:
            feats.append(low_template_correlation)
        return np.array([feats]), np.array([rec.labels[0]]), rec.group

    def select_features(self, X, y, cfg=None):
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        cfg = self._cfg(cfg)

        # First create and fit an imputer on the training fold
        imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )
        X_imputed = imputer.fit_transform(X)

        # Fit the normal feature selector on imputed training data
        selector = super().select_features(X_imputed, y, cfg=cfg)

        # Return a fitted pipeline so evaluate() can transform
        # validation recordings using the same imputer
        return Pipeline([
            ("imputer", imputer),
            ("selector", selector)
        ])


    def baseline(self, cfg=None):
        """Module 5 — the supplied random forest. `cfg["imbalance"]` selects how
        class imbalance is handled ("none" | "balanced" | "balanced_subsample" |
        "resample" | "smote" | "adasyn" | "threshold"); the trade-off table lives
        on `adapter.default_baseline`. It is a *choice you report*, not a default
        nobody sees — the rubric grades it.

        `"smote"`/`"adasyn"` synthesise minority rows by INTERPOLATING between
        real neighbours instead of duplicating them (`"resample"`) or reweighting
        the loss (`"balanced"`) — fold-safe via `adapter.SMOTEd`, and needing the
        optional `imbalanced-learn` package. Before you use it, ask what a point
        halfway between two of THIS track's epochs would physically be."""

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        cfg = self._cfg(cfg)

        classifier = cfg.get("classifier")

        if classifier == "knn":
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=cfg.get("n_neighbors", 5),
                    weights=cfg.get("weights", "uniform")
                ))
            ])
        elif classifier == "svm":
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", SVC(
                    kernel = cfg.get("kernel", "rbf"),
                    C = cfg.get("C", 1.0),
                    gamma = cfg.get("gamma", "scale"),
                    class_weight=cfg.get("class_weights", "balanced")
                ))
            ])
        elif classifier == "rf":
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=cfg.get("n_estimators", 200),
                    max_depth=cfg.get("max_depth", None),
                    min_samples_leaf=cfg.get("min_samples_leaf", 1),
                    class_weight=cfg.get("class_weight", "balanced"),
                    random_state=cfg.get("seed", 0)
                ))
            ])
        else:
            print(f"Unknown classifierm{classifier}, using defualt baseline instead")
            pipe = default_baseline(seed=int(cfg.get("seed", 0)),
                                            imbalance=cfg.get("imbalance", "balanced"),
                                            threshold=float(cfg.get("threshold", 0.5)),
                                            smote_k=int(cfg.get("smote_k", 5)))
        return pipe        


        

    _PN_BASE = "https://physionet.org/files/challenge-2017/1.0.0/training/"

    def download(self, cache_dir, subset=400, seed=0, workers=8, exclude_ids = None):
        """Fetch a class-stratified subset of the CinC-2017 training set into
        cache_dir. Needs network; `wfdb` not required for the download itself.

        The full training set is ~600 MB (8,528 records); each record is only a
        ~18 KB single-lead ECG, so we pull just `subset` records — stratified
        across N/A/O/~ with a floor per class so the rare classes (AF, Noisy)
        stay learnable. Files land flat in cache_dir as `A0xxxx.hea`/`.mat`
        alongside `REFERENCE-v3.csv`.
        """
        import os, urllib.request, urllib.error, concurrent.futures as cf
        os.makedirs(cache_dir, exist_ok=True)
        ref_dst = os.path.join(cache_dir, "REFERENCE-v3.csv")
        if not os.path.exists(ref_dst):
            urllib.request.urlretrieve(self._PN_BASE + "REFERENCE-v3.csv", ref_dst)
        # group record ids (with their A0x/ subfolder prefix) by class label
        by_cls = {c: [] for c in self.meta.classes}
        if exclude_ids is None:
            exclude_ids = set()

        for line in open(ref_dst):
            line = line.strip()
            if "," not in line:
                continue
            rid, lab = line.split(",", 1)
            if os.path.basename(rid) in HOLDOUT_RECORDS:      # hold-out records are never part of training
                continue
            if os.path.basename(rid) in exclude_ids:      # exclude chosen ID:s
                            continue
            by_cls.setdefault(lab, []).append(rid)
        # stratified pick: proportional share of `subset`, but >= floor per class
        rng = np.random.default_rng(seed)
        floor = 40
        total_ref = sum(len(v) for v in by_cls.values())
        picks = []
        for c in self.meta.classes:
            ids = by_cls.get(c, [])
            want = max(floor, round(subset * len(ids) / max(total_ref, 1)))
            want = min(want, len(ids))
            idx = rng.choice(len(ids), size=want, replace=False)
            picks.extend(ids[i] for i in idx)

        def _grab(rid_full):
            base = os.path.basename(rid_full)               # 'A00/A00001' -> 'A00001'
            for ext in (".hea", ".mat"):
                dst = os.path.join(cache_dir, base + ext)
                if os.path.exists(dst) and os.path.getsize(dst) > 0:
                    continue
                for attempt in range(3):
                    try:
                        urllib.request.urlretrieve(self._PN_BASE + rid_full + ext, dst)
                        break
                    except (urllib.error.URLError, OSError):
                        if attempt == 2:
                            raise
            return base

        got = 0
        total = len(picks)
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(_grab, picks):
                got += 1
                print(
                f"\r[ecg_cinc2017] Progress: {got}/{total} records",
                end="\n" if got == total else "",
                flush=True
                )
        print(f"[ecg_cinc2017] downloaded {got} records into {cache_dir}")
        return cache_dir

    def download_holdout(self, cache_dir, workers=8):
        """Fetch ONLY the hold-out records into cache_dir. Use a separate folder from your training
        data, and only for the final prediction (see HOLDOUT_EVALUATION.md)."""
        import os, urllib.request, concurrent.futures as cf
        os.makedirs(cache_dir, exist_ok=True)
        ref_dst = os.path.join(cache_dir, "REFERENCE-v3.csv")
        if not os.path.exists(ref_dst):
            urllib.request.urlretrieve(self._PN_BASE + "REFERENCE-v3.csv", ref_dst)
        full = {os.path.basename(l.split(",")[0]): l.split(",")[0] for l in open(ref_dst) if "," in l}

        def _grab(rid):
            for ext in (".hea", ".mat"):
                dst = os.path.join(cache_dir, rid + ext)
                if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
                    urllib.request.urlretrieve(self._PN_BASE + full[rid] + ext, dst)
            return rid

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            got = list(ex.map(_grab, HOLDOUT_RECORDS))
        print(f"[ecg_cinc2017] downloaded {len(got)} hold-out records into {cache_dir}")
        return cache_dir

    def load(self, cache_dir):
        """Read CinC-2017 WFDB records + REFERENCE labels; one Recording per record.
        Robust to both the flat layout (records copied into cache_dir) and the
        nested `A0x/` layout from unzipping training2017.zip; REFERENCE ids may
        carry an `A0x/` prefix, so we match on the basename."""
        import os, csv, glob
        import wfdb
        ref = {}
        for name in ("REFERENCE-v3.csv", "REFERENCE.csv"):
            p = os.path.join(cache_dir, name)
            if os.path.exists(p):
                for row in csv.reader(open(p)):
                    if len(row) >= 2:
                        ref[os.path.basename(row[0])] = row[1]
                break
        recs = []
        for hea in sorted(glob.glob(os.path.join(cache_dir, "**", "*.hea"), recursive=True)):
            rid = os.path.splitext(os.path.basename(hea))[0]
            if rid not in ref:
                continue
            rec = wfdb.rdrecord(os.path.splitext(hea)[0])
            x = rec.p_signal[:, 0].astype(float)
            recs.append(Recording(group=rid, fs=float(rec.fs),
                                  epochs={"ecg": x[None, :]}, labels=np.array([ref[rid]])))
        return recs

