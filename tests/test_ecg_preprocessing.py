"""Regression tests for the T10 ECG preprocessing configuration."""
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks"))

from ecg_cinc2017 import ECGCinC2017Track  # noqa: E402


SOFIA_PREPROCESSING_CFG = {
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
}


def _one_smoke_record(track):
    return track.smoke(per_class=1, fs=300, dur=5.0)[0]


def test_documented_numeric_bandpass_still_works():
    track = ECGCinC2017Track(cfg={"preprocess": "bandpass", "ecg_band": (0.5, 40.0)})
    rec = _one_smoke_record(track)
    out = track.preprocess(rec)

    assert out.epochs["ecg"].shape == rec.epochs["ecg"].shape
    assert np.isfinite(out.epochs["ecg"]).all()


def test_combined_preprocessing_runs_without_mutating_input():
    track = ECGCinC2017Track(cfg=SOFIA_PREPROCESSING_CFG)
    rec = _one_smoke_record(track)
    before = rec.epochs["ecg"].copy()

    out = track.preprocess(rec)

    assert out.epochs["ecg"].shape == before.shape
    assert np.isfinite(out.epochs["ecg"]).all()
    assert np.array_equal(rec.epochs["ecg"], before)
    assert not np.allclose(out.epochs["ecg"], before)
