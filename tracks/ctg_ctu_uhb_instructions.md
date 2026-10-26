# Track instructions — CTG (fetal), CTU-UHB

> ⚠️ **Not offered this year.** This track isn't one of the options for CM2013 HT26 — its honest
> baseline sits far below the other tracks, closer to a floor than a moderate challenge. See
> [`START_HERE.md`](START_HERE.md) for the tracks actually on offer. Kept here as reference / for
> future years.

*Attach the dataset card (`ctg_ctu_uhb_card.md`). Adapter: `ctg_ctu_uhb.py`.*

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "CTG / fetal"
> section): it lists the **book sections** (dropout/interpolation §12.9–12.11, baseline/variability,
> deceleration detection via the Pan–Tompkins template §9.9 …), the **literature-review anchors** (CTU-UHB
> paper, **FIGO** guidelines defining deceleration types), and the **guiding questions** your report must
> answer to *motivate* your design (course outcome **L5**).

## 1. The problem
From an intrapartum cardiotocogram — **fetal heart rate (FHR)** + **uterine contractions (UC)** —
predict whether the delivery outcome was **normal** or **pathological** (fetal acidemia). One label
per recording, derived from umbilical-artery **pH**.

## 2. Your data (see the dataset card)
- Dataset: **CTU-UHB Intrapartum CTG** (ODC-BY, direct download — **no agreement**).
- Signals: FHR + UC (toco), both **4 Hz**, ~90 min/record. Labels: normal / pathological (pH < 7.15).
- **Split unit (leakage): recording** — one label per record; never leak a record across folds.
- Smoke subset for setup/CI: records `1001, 1002, 1004`. Full subset runtime: ~2–4 min.
- Evaluation mode: **new-recording** (GroupKFold on record id).

## 3. What you are given (do not rebuild these)
- The **modular pipeline** and a **working baseline** (`tracks.adapter.default_baseline`).
- The adapter (`ctg_ctu_uhb.py`) with `smoke()`, `preprocess()` (inherited identity — **not** a
  `cfg`-wired stage on this track, so `cfg={"preprocess": ...}` is rejected; dropout handling
  belongs here once you choose a strategy, but you get there by **overriding the method**, not
  by passing a cfg key — see §4c), `extract_features()` (dropout cleaning + baseline/variability
  + accel/decel + UC features on the **last 30 min**), and a `wfdb` `download()/load()` that parses pH.
- The shared leakage-safe **evaluator** (`TrackAdapter.evaluate`, GroupKFold by record).
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across folds** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** on the smoke subset, then on real records. It barely detects the pathological
   class (recall ≈ 0.1) — that is your starting point, not your result.
2. **Improve the signal processing**: better dropout handling; deceleration morphology (late vs
   variable vs early), deceleration area, short-term variability (STV) done properly, sample entropy;
   try the **last 15 min** or contraction-locked windows.
3. **Improve the model / threshold** (validated inside folds); consider the pH threshold's effect.
4. **Analyse failures** — which pathological records are missed, and what does their trace look like?

Those steps are a *menu, not a march*: which preprocessing, which features, whether to select
features at all, and which learner to climb to are decisions the scaffold deliberately leaves
open (see `preprocess()` and `make_selector()` / `default_baseline()` in `adapter.py` for the options and their
trade-offs). The grade is on the quality of the reasoning, not on matching one blessed recipe.

## 4c. The design-decision menus (stages 2-5) — what the scaffold offers, and the trade-off

> **The option menus themselves — the Chapter-8 denoise-by-noise-type table, spectral estimators,
> feature selection, class imbalance, and why the validation split is *not* a choice — are in
> [`DESIGN_MENUS.md`](DESIGN_MENUS.md), written once for all tracks. Everything below is what those
> choices mean **on this track**.**

### Stage 2 — preprocessing · the dropout decision, shipped in the wrong place

`_clean_fhr()` is called from `extract_features()` for convenience, but it is **preprocessing**.
Moving it into `preprocess()` and choosing what to do with the signal-loss gaps — interpolate,
exclude, or keep-and-flag — is one of the decisions you will need to defend in your report.
`adapter.denoise()` is importable and gives you the Chapter 8 per-noise-type menu in [`DESIGN_MENUS.md`](DESIGN_MENUS.md) (call it from a `preprocess()` you write — it is not a `cfg` key on this track); note that raw FHR dropout is a
**data-integrity** problem in §8.2's sense ("bad segments should be detected and excluded or flagged,
**not silently filtered**"), not a noise-remedy problem, and conflating the two is the specific
mistake to avoid here: it costs you under Criterion 2 (signal-processing rigor).

### Stage 2 — the Chapter 8 noise-type menu · **available as functions, not wired into this track's `cfg`**

`CTGTrack` does not wire `preprocess` into `cfg` at all — dropout handling (`_clean_fhr`) is
called from `extract_features()` by default, which is exactly the misplacement this track asks
you to argue about — so `cfg={"preprocess": "denoise", ...}` is **rejected** with an
`UnsupportedCfgKey` error rather than silently ignored. Moving it is a code change (override
`preprocess()` yourself, as §3 says), not a cfg toggle.

The functions themselves are real and importable, and the per-noise-type table in [`DESIGN_MENUS.md`](DESIGN_MENUS.md) is the decision you should still be making — you just have to **call them yourself** from a `preprocess()` you write (`from adapter import denoise, bandpass_notch, wavelet_denoise`). Once you have, register the knobs so they become first-class config options:

```python
track = CTGTrack()
track.declare_cfg_keys("preprocess", "impulsive", "baseline", "powerline", "broadband")
```

### Stage 3 — spectral estimation · available, but **not wired into this track's features**

`adapter.make_spectral_estimator` / `spectral_bandpower` offer the Chapter 7 menu (`"periodogram"` ·
`"bartlett"` · `"welch"` · `"multitaper"` · `"ar"`), but this track's shipped `extract_features()`
computes no PSD-integrated band powers, so `cfg["spectral_method"]` has nothing to act on here.

The variability features (STV/LTV) are windowed variances in the time domain, not band powers.

## 5. Deliverables
- A short **report** (features justified physiologically; results vs. the class-prior baseline).
- `predictions.csv` (`record,label`) on the held-out split for hold-out evaluation.
- A slot in the **cross-track showcase**: "same variability/deceleration ideas, our signal."
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Compare against the supplied baseline **honestly** — beating it is not required; a defended
  decision to keep a lower-scoring pipeline earns full marks (Criterion 7). State the **split
  unit (recording)**, the **pH threshold**, and the evaluation mode with every number. Never report smoke/CI numbers as results. Report the metric **with its spread** across folds (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: raw FHR is full of signal-loss zeros/spikes (clean first), strong class imbalance, the
label is a pH threshold you must state, and the acidemia signal concentrates near delivery (window the tail).
