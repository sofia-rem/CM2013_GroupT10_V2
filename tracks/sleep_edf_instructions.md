# Track instructions — Sleep staging, Sleep-EDF

*Attach the dataset card (`sleep_edf_card.md`). Adapter: `sleep_edf.py`. **Reference track.***

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "Sleep
> staging" section): it lists the **book sections** that cover each method (a refresher — you already
> learned these), the **literature-review anchors** (dataset paper, AASM scoring standard, benchmark
> models), and the **guiding questions** your report must answer to *motivate* your design (course
> outcome **L5**).

## 1. The problem
From overnight PSG — **EEG + EOG + chin EMG** — assign a **sleep stage** (W / N1 / N2 / N3 / REM) to
every 30-second epoch, producing a hypnogram.

## 2. Your data (see the dataset card)
- Dataset: **Sleep-EDF Expanded** (ODC-BY, direct download via `mne` — **no agreement**).
- Signals: EEG Fpz-Cz + Pz-Oz, EOG, EMG; resampled to 100 Hz. Labels: 5 AASM stages (R&K→AASM merge).
- **Split unit (leakage): subject** — Sleep-Cassette has **two nights per subject**; splitting by night
  leaks subject identity. Never let a subject appear in train and test.
- Smoke subset: `SC4001, SC4011, SC4021`. Evaluation mode: **new-subject** (LOSO).

## 3. What you are given (do not rebuild these)
- The modular pipeline + a working baseline (`tracks.adapter.default_baseline`).
- The adapter (`sleep_edf.py`) with `smoke()`, an opt-in `preprocess()` (band-pass/notch — off by
  default, yours to choose), `extract_features()` (band power + Hjorth + entropy + EOG/EMG), and a
  Colab `download()/load()` that does the R&K→AASM merge and drops Movement/Unknown.
- The shared leakage-safe **evaluator** (`TrackAdapter.evaluate`, leave-one-subject-out).
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across subjects** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** on the smoke subset, then real nights; report the honest panel (κ, macro-F1, confusion).
2. **Improve the signal processing / features**: spindle detection (STFT/wavelet), better band-power
   estimation (multitaper), EOG/EMG artifact handling; consider cropping to lights-off ± 30 min.
3. **Improve the model / selection** (validated inside folds).
4. **Analyse failures** — N1 is the hard class; where do W↔N1↔N2 and N2↔N3 errors happen?

Each of those steps is a *menu*, not a march: the band you filter to, whether spindles come from an
STFT or a wavelet, whether artifacts are rejected / interpolated / flagged, whether you select features
at all — the scaffold ships options and their trade-offs (see `preprocess()` and `make_selector()` / `default_baseline()` in
`adapter.py`), and the grade is on the quality of your reasoning, not on matching one blessed recipe.

## 4b. Reporting (module 7) — what to show, and the code that draws it

```python
import report as R                       # tracks/report.py

rep = track.evaluate(X, y, groups)
track.report(rep)                        # confusion matrix first, then κ WITH its spread
R.plot_hypnogram(y_true_night, y_pred_night)          # predicted vs. reference, disagreements marked
R.compare_stage_summaries(y_true_night, y_pred_night) # TST, sleep efficiency, WASO, SOL, REM latency
```

- Lead with the **confusion matrix** and κ, never bare accuracy (§16.8).
- Quote κ **with its spread across subjects** — "mean κ 0.61 (range 0.34–0.73 across 8 subjects)" —
  because one hard sleeper is exactly what a pooled number hides.
- Show one **hypnogram**: a good per-epoch κ can still get the night's *shape* wrong.
- `compare_stage_summaries` states its own conventions (time in bed = the scored sequence, onset =
  first non-wake epoch); if your clinical definitions differ, use them and say which. With several
  nights, plotting the difference column against the mean gives you the Bland–Altman view of §16.5.

## 4c. The design-decision menus (stages 2-5) — what the scaffold offers, and the trade-off

> **The option menus themselves — the Chapter-8 denoise-by-noise-type table, spectral estimators,
> feature selection, class imbalance, and why the validation split is *not* a choice — are in
> [`DESIGN_MENUS.md`](DESIGN_MENUS.md), written once for all tracks. Everything below is what those
> choices mean **on this track**.**

### Stage 2 — preprocessing · `cfg["preprocess"]`

`"none"` (default) · `"bandpass"` · `"wavelet"` · `"denoise"`. Setting `eeg_band`/`notch` alone still
turns the band-pass on, so older configs keep working.

**`"bandpass"` vs `"wavelet"` answer different noise signatures.** A fixed band is right for
*stationary, narrow-band* interference — 50 Hz hum sits in one place all night — but it is applied to
every instant equally, so it **smears the sharp transients**: K-complexes and spindle onsets are
brief broadband events and band-limiting rounds their edges. Wavelet thresholding asks instead which
*coefficients are too large to be noise*, localised in time as well as scale, so it removes movement
arousals and electrode pops (which live in no single band) while leaving the sharp K-complex edge
intact — at the cost of eroding **weak spindles** if the threshold is set too high.

### Stage 2 — preprocessing, **indexed by noise type** · `cfg["preprocess"] = "denoise"`

On this track it bites hardest because the bands are narrow (the book's own edges: delta 0.5-4,
theta 4-8, alpha 8-11, sigma 11-16, beta 16-30 Hz). Once the practical resolution coarsens past a
few Hz you are no longer measuring the band you named — alpha and sigma start reporting each other's
power. Only the five EEG band powers are recomputed; Hjorth, entropy, EOG and EMG are untouched, so
any metric change is the estimator's doing.

## 5. Deliverables
- A short **report** reading your κ against the **human ceiling (κ ≈ 0.76)** — a number *above* it is a
  red flag (Wake imbalance / tiny subset), not a win. Justify every design choice.
- `predictions.csv` (`record,epoch,label`) on the held-out split for hold-out evaluation.
- A cross-track showcase slot: "the same band-power/feature ideas, on EEG."
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Compare against the supplied baseline **honestly** — beating it is not required; a defended
  decision to keep a lower-scoring pipeline earns full marks (Criterion 7). State the **split
  unit (subject)** and evaluation mode with every number. Never report smoke/CI numbers as results. Report the metric **with its spread** across subjects (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: R&K→AASM merge (S3+S4→N3, drop Movement/Unknown), **split by subject not night**, Wake
dominates real recordings (accuracy is inflated — read κ), and N1 is rare and hard.
