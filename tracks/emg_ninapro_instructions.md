# Track instructions — EMG gesture, Ninapro DB1

*Attach the dataset card (`emg_ninapro_card.md`). Adapter: `emg_ninapro.py`.*

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "EMG gesture"
> section): it lists the **book sections** (rectified/analytic envelope §5.5, time-domain features §11.3,
> within- vs cross-subject domain shift §12.3/§12.13 …), the **literature-review anchors** (Ninapro DB1
> paper, the **Hudgins** feature set), and the **guiding questions** your report must answer to *motivate*
> your design (course outcome **L5**).

## 1. The problem
From **10-channel surface EMG** of the forearm, classify which **hand gesture** is being performed
(exercise E1 = 12 finger movements). One label per short window.

## 2. Your data (see the dataset card)
- Dataset: **Ninapro DB1** (direct per-subject zip download — **no login/agreement**; cite Atzori 2014).
- Signals: 10-channel sEMG **envelope @ 100 Hz** (already rectified/low-passed). Labels: 12 movements.
- **Split unit (leakage): subject** (new-subject) or **repetition** (within-subject) — never let windows
  from the same repetition span train and test.
- Smoke subset: subjects `S1, S2, S3`. Two evaluation modes: **within-subject** and **new-subject**.

## 3. What you are given (do not rebuild these)
- The modular pipeline + a working baseline (`tracks.adapter.default_baseline`).
- The adapter (`emg_ninapro.py`) with `smoke()`, `preprocess()` (identity by default), 
  `extract_features()` (MAV/RMS/WL/VAR/MNF per channel, windowed), `download()/load()`, and
  **`evaluate_modes()`** that reports within- and new-subject side by side, each with its
  per-subject spread.
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across subjects** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** in both modes. Note the gap: ~0.78 within-subject vs ~0.19 new-subject.
2. **Improve the signal processing / features**: better windowing, Hilbert envelope, per-channel
   normalisation, channel-relative features; consider frequency features (but remember DB1 is an envelope).
3. **Attack the cross-subject gap**: per-subject calibration, normalisation, or domain adaptation.
4. **Analyse failures** — which gestures/channels confuse, and why (anatomy, electrode shift)?

Those steps are a *menu, not a march*: which preprocessing, which features, whether to select
features at all, and which learner to climb to are decisions the scaffold deliberately leaves
open (see `preprocess()` and `make_selector()` / `default_baseline()` in `adapter.py` for the options and their
trade-offs). The grade is on the quality of the reasoning, not on matching one blessed recipe.

## 4c. The design-decision menus (stages 2-5) — what the scaffold offers, and the trade-off

> **The option menus themselves — the Chapter-8 denoise-by-noise-type table, spectral estimators,
> feature selection, class imbalance, and why the validation split is *not* a choice — are in
> [`DESIGN_MENUS.md`](DESIGN_MENUS.md), written once for all tracks. Everything below is what those
> choices mean **on this track**.**

### Stage 2 — preprocessing · a stub you fill in

DB1 ships a rectified, low-pass **envelope**, so the usual raw-sEMG moves (zero crossings, slope-sign
changes, a fresh band-pass) are meaningless here — knowing that is half the exercise. What belongs in
this stage is per-channel normalisation *within a subject*, envelope smoothing, window length and
channel re-ordering. `adapter.denoise()` is importable and gives you the Chapter 8 per-noise-type menu in [`DESIGN_MENUS.md`](DESIGN_MENUS.md) — but you must call it from a `preprocess()` you write; it is not a `cfg` key on this track.

### Stage 2 — the Chapter 8 noise-type menu · **available as functions, not wired into this track's `cfg`**

`EMGNinaproTrack.preprocess()` is an identity stub (DB1 already ships a rectified, low-pass envelope), so it forwards no denoise keys and `cfg={"preprocess": "denoise", ...}` is **rejected** with an `UnsupportedCfgKey` error rather than silently ignored.

The functions themselves are real and importable, and the per-noise-type table in [`DESIGN_MENUS.md`](DESIGN_MENUS.md) is the decision you should still be making — you just have to **call them yourself** from a `preprocess()` you write (`from adapter import denoise, bandpass_notch, wavelet_denoise`). Once you have, register the knobs so they become first-class config options:

```python
track = EMGNinaproTrack()
track.declare_cfg_keys("preprocess", "impulsive", "baseline", "powerline", "broadband")
```

### Stage 3 — spectral estimation · available, but **not wired into this track's features**

`adapter.make_spectral_estimator` / `spectral_bandpower` offer the Chapter 7 menu (`"periodogram"` ·
`"bartlett"` · `"welch"` · `"multitaper"` · `"ar"`), but this track's shipped `extract_features()`
computes no PSD-integrated band powers, so `cfg["spectral_method"]` has nothing to act on here.

The one spectral feature per channel is a mean frequency read off a raw `|rFFT|` magnitude spectrum —
which is worth examining in light of §7.8's "common student mistakes" box, since a raw magnitude
spectrum is exactly the high-variance estimate Chapter 7 warns about. Replacing it with a proper
integrated-density estimate is a legitimate stage-3 improvement to propose and measure.

## 5. Deliverables
- A short **report** stating the **evaluation mode** for every number (within- vs new-subject).
- `predictions.csv` on the held-out split for hold-out evaluation.
- A cross-track showcase slot: "same energy/frequency features, our signal — what broke cross-subject?"
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Compare against the supplied baseline **honestly, per mode** — beating it is not required; a
  defended decision to keep a lower-scoring pipeline earns full marks (Criterion 7). Never mix
  modes or report within-subject as if it were new-subject. State the split unit with every number. Report the metric **with its spread** across subjects (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: DB1 is a **rectified envelope** (no zero-crossing/SSC features), cross-subject sEMG is
near-useless without adaptation, and window-overlap across the split unit leaks.
