# Track instructions — ECG rhythm / AF, PhysioNet CinC-2017

*Attach the dataset card (`ecg_cinc2017_card.md`). Adapter: `ecg_cinc2017.py`.*

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "ECG rhythm /
> AF" section): it lists the **book sections** (Pan–Tompkins §9.9, HRV §7.11, notch §9.6 …), the
> **literature-review anchors** (Challenge paper, Pan & Tompkins 1985, HRV Task Force), and the
> **guiding questions** your report must answer to *motivate* your design (course outcome **L5**).

## 1. The problem
From a **single-lead ECG** (300 Hz, ~10–60 s), classify the rhythm as **Normal / AF / Other / Noisy** —
the task a wearable rhythm monitor must do. One label per recording.

## 2. Your data (see the dataset card)
- Dataset: **PhysioNet/CinC-2017** (open, direct download via `wfdb` — **no agreement**).
- Signals: single-lead ECG @ 300 Hz, variable length. Labels: N / A / O / ~ (strongly imbalanced).
- **Split unit (leakage): record** — one label per recording; GroupKFold on record id.
- Smoke subset: `A00001, A00004, A00006`. Evaluation mode: **new-record**.

## 3. What you are given (do not rebuild these)
- The modular pipeline + a working baseline (`tracks.adapter.default_baseline`).
- The adapter (`ecg_cinc2017.py`) with `smoke()`, `preprocess()` (identity by default — hoisting the
  band-pass out of `_rpeaks` is one option), `extract_features()` (QRS detection → R–R/HRV + a
  signal-quality index), a stratified `download()` and a `wfdb` `load()`.
- The shared leakage-safe **evaluator** (GroupKFold by record).
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across folds** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** on the smoke subset, then real records; report macro-F1, κ, per-class recall.
2. **Improve the signal processing / features**: robust QRS detection under noise, better R–R
   irregularity/HRV features, and an explicit **signal-quality** feature set for the Noisy class.
3. **Improve the model / selection** (validated inside folds); handle the class imbalance deliberately.
4. **Analyse failures** — where do A↔O confuse, and which records are mislabelled "Noisy"?

Those steps are a *menu, not a march*: which preprocessing, which features, whether to select
features at all, and which learner to climb to are decisions the scaffold deliberately leaves
open (see `preprocess()` and `make_selector()` / `default_baseline()` in `adapter.py` for the options and their
trade-offs). The grade is on the quality of the reasoning, not on matching one blessed recipe.

## 4c. The design-decision menus (stages 2-5) — what the scaffold offers, and the trade-off

> **The option menus themselves — the Chapter-8 denoise-by-noise-type table, spectral estimators,
> feature selection, class imbalance, and why the validation split is *not* a choice — are in
> [`DESIGN_MENUS.md`](DESIGN_MENUS.md), written once for all tracks. Everything below is what those
> choices mean **on this track**.**

### Stage 2 — preprocessing · `cfg["preprocess"]`

`"none"` (default) · `"bandpass"` · `"wavelet"` · `"denoise"`. The shipped default lets `_rpeaks()`
own its own band-pass and lets every other feature see the raw strip; `"bandpass"` hoists the filter
into `preprocess()` so it becomes a reportable, swappable stage.

**The catch that makes this track distinctive: denoising can destroy the label.** Two of the eleven
shipped features (the QRS/high-frequency ratio and the 49-51 Hz powerline power) exist to *measure*
corruption, and the `~` class **is** the low-quality class. Any cleaning applied before those
features are computed makes Noisy harder to detect, not easier. Separately, the **QRS complex is a
broadband ~80 ms transient**, so a fixed band-pass rounds its edges, widens its apparent duration and
shifts the peak your R-R intervals are measured from — which is exactly the case `"wavelet"` exists
to handle. Measure it both ways and report which you chose.

### Stage 2 — preprocessing, **indexed by noise type** · `cfg["preprocess"] = "denoise"`

Three of the eleven features are band powers. The 49-51 Hz powerline feature is the narrowest band in
the whole scaffold — 2 Hz wide — so it is the one most punished by a coarsened estimator and most
helped by `"multitaper"` on a short strip. Conversely `"ar"` will happily place a pole near 50 Hz and
report a confident, smooth, possibly fictitious hum peak: §7.9's over-fitting warning arriving as a
feature value. A 20-30 s strip at 300 Hz is long enough for Welch to be comfortable, so expect small
movements — say so honestly if that is what you measure.

## 5. Deliverables
- A short **report** reading your macro-F1 against the **Challenge winners (~0.83)**. Justify choices.
- `predictions.csv` (`record,label`) on the held-out split for hold-out evaluation.
- A cross-track showcase slot: "Pan–Tompkins event detection, on ECG."
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Compare against the supplied baseline **honestly** — beating it is not required; a defended
  decision to keep a lower-scoring pipeline earns full marks (Criterion 7). State the **split
  unit (record)** with every number. Never
  report smoke/CI numbers as results. Report the metric **with its spread** across folds (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: single lead + variable length, the **Noisy class is a signal-quality problem** (not a
rhythm), strong imbalance (score macro-F1/κ), and a bad QRS detector wrecks every downstream feature.
