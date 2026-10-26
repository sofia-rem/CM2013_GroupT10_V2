# Track instructions — BCI motor imagery, EEGMMIDB

*Attach the dataset card (`bci_eegmmidb_card.md`). Adapter: `bci_eegmmidb.py`. **Advanced track — opt-in only this year, instructor sign-off required; not in the default assignment pool.***

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "BCI motor
> imagery" section): it lists the **book sections** (SSVEP-BCI averaging §10.4, mu/beta ERD §7.5, spatial
> unmixing PCA/ICA→CSP §10.7 …), the **literature-review anchors** (BCI2000 paper, ERD/ERS, **CSP**), and
> the **guiding questions** your report must answer to *motivate* your design (course outcome **L5**).

## 1. The problem
From **64-channel EEG** during motor imagery, decide whether the subject imagined moving the
**left or right hand** (runs R04/R08/R12). One label per ~4 s imagery trial.

## 2. Your data (see the dataset card)
- Dataset: **EEGMMIDB** (ODC-BY, direct download via `mne.datasets.eegbci` — **no agreement**).
- Signals: 64-ch EEG **@ 160 Hz**. Labels: L / R (annotation T1 = left fist, T2 = right fist).
- **Split unit (leakage): subject** (new-subject) or **trial** (within-subject).
- Smoke subset: `S001, S002, S003`. Two evaluation modes: **within-subject** and **new-subject**.

## 3. What you are given (do not rebuild these)
- The modular pipeline + a working baseline (`tracks.adapter.default_baseline`).
- The adapter (`bci_eegmmidb.py`) with `smoke()`, `preprocess()` (identity by default — **not** a
  safe home for a learned spatial filter: it runs once over every recording *before* `evaluate()`
  cuts the CV folds, so anything it learns has already seen every subject. The one place a
  learned CSP filter is genuinely fit inside the fold is the `clf=` Pipeline — see §4 step 2),
  `extract_features()`
  (log mu/beta band power over a sensorimotor montage + C3–C4 laterality), `download()/load()`, and
  **`evaluate_modes()`** — each mode reported with its per-subject spread.
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across subjects** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** in both modes. It is **near chance** — that is your starting point and the
   whole challenge, not a bug.
2. **Add a spatial filter**: **CSP** (Common Spatial Patterns) fit **inside each CV fold**, then LDA —
   this is the single biggest lever. Tune subject-specific mu/beta bands. See the worked example
   in §4c (Stage 2) for the one place in this scaffold a learned CSP filter can actually live
   without leaking.
3. **Handle non-stationarity** for the new-subject mode (normalisation, transfer).
4. **Analyse failures** — which subjects are decodable at all? (Many EEGMMIDB subjects are not.)

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

The moves that matter here are a **CSP spatial filter fit inside the training fold**, a
subject-specific mu band, and a Laplacian / common-average re-reference. `adapter.denoise()` is importable (call it from a `preprocess()` you write; it is not a `cfg` key on this track) and gives you
the Chapter 8 per-noise-type menu in [`DESIGN_MENUS.md`](DESIGN_MENUS.md) if what you actually see in the traces is hum, drift or pops
rather than a spatial-mixing problem.

**Where CSP actually goes.** `preprocess()`/`extract_features()` both run once over *every*
recording before `evaluate()` cuts the CV folds (`adapter.build_dataset.__doc__`); a CSP filter
written into either one has already seen the held-out subjects' data by the time a fold is
built — a leak `assert_no_subject_leak` cannot catch, because it isn't a group-id problem. And
by the time `select_features()` runs, `extract_features()` has already reduced each trial to a
handful of band-power numbers, so there is no multichannel signal left for CSP to work on
there either. The one place a *learned* spatial filter is genuinely fit fresh inside every
fold is the classifier pipeline you pass as `clf=` — `evaluate()`/`evaluate_modes()` `clone()`
it per fold. That means the CSP route needs two changes together, not one:

```python
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class MyBCITrack(BCIEEGMMIDBTrack):
    def extract_features(self, rec, cfg=None):
        # CSP needs the raw, per-trial, multichannel array -- return that instead
        # of collapsing to band powers, and let the clf= pipeline do the rest.
        return rec.epochs["eeg"], rec.labels, rec.group   # [n_trials, n_ch, n_samp]

track = MyBCITrack()
clf = Pipeline([("csp", CSP(n_components=6)), ("lda", LinearDiscriminantAnalysis())])
X, y, g = track.build_dataset(recordings)                  # now raw trials, not band powers
rep = track.evaluate(X, y, g, clf=clf)                      # CSP is refit fresh, per fold
```
State in `RESULTS.md` that you did this and why — "we fit CSP inside the fold via `clf=`,
after overriding `extract_features()` to keep the raw trial array" is the sentence Criterion 3
is looking for.

### Stage 2 — the Chapter 8 noise-type menu · **available as functions, not wired into this track's `cfg`**

`BCIEEGMMIDBTrack.preprocess()` is an identity stub — the stage-2 move with the literature behind it here is a CSP spatial filter you write and fit inside the training fold — so it forwards no denoise keys, and `cfg={"preprocess": "denoise", ...}` is **rejected** with an `UnsupportedCfgKey` error rather than silently ignored.

The functions themselves are real and importable, and the per-noise-type table in [`DESIGN_MENUS.md`](DESIGN_MENUS.md) is the decision you should still be making — you just have to **call them yourself** from a `preprocess()` you write (`from adapter import denoise, bandpass_notch, wavelet_denoise`). Once you have, register the knobs so they become first-class config options:

```python
track = BCIEEGMMIDBTrack()
track.declare_cfg_keys("preprocess", "impulsive", "baseline", "powerline", "broadband")
```

**This is the track Chapter 7 §7.6-7.7 were written for.** Every feature here is a band power, and a
trial is 4 s minus a 0.5 s trim at 160 Hz — about 560 samples. Welch has to chop that into a handful
of segments and quickly "runs out of segments to average"; §7.16 recommends multitaper for exactly
this case ("short record, want low-leakage smooth PSD (EEG/neuro)") and §7.7 recommends AR for "short
segment, resonant rhythm", which is a literal description of the mu rhythm. Single-trial band power
is the noisiest quantity in the scaffold, so a better estimator has more room to help here than
anywhere else — and `"ar"` is correspondingly most dangerous: at too high an order one of §7.9's
spurious peaks landing in 8-12 Hz becomes a confident, fictitious ERD.

## 5. Deliverables
- A short **report** with the **evaluation mode** on every number, and an honest account of which
  subjects were decodable.
- `predictions.csv` on the held-out split for hold-out evaluation.
- A cross-track showcase slot: "spatial filtering was the difference."
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Compare against the supplied baseline **honestly, per mode** — beating it is not required; a
  defended decision to keep a lower-scoring pipeline earns full marks (Criterion 7). **Fit
  CSP/any spatial filter inside the fold** — fitting on all data is a leak that fakes high scores. State the split unit with every number. Report the metric **with its spread** across subjects (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: near-chance naive baseline (needs CSP), spatial-filter leakage if fit outside folds,
brutal cross-subject non-stationarity, and the T1/T2 = left/right label semantics.
