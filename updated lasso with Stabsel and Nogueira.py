"""
Reproduction of Figure 3 (page 6) from:
  "Selection and Stability of Functional Connectivity Features for
   Classification of Brain Disorders" — Saha, Hazra & Ghosh (2025)

Loads: /kaggle/input/datasets/kinnarhalder/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat
Outputs:
  - fig3_combined.png         (2×3 grid: Accuracy, F1, KI, JI, Nogueira, Nogueira vs KI)
  - fig3_accuracy.png, fig3_f1score.png, fig3_kuncheva.png,
    fig3_jaccard.png, fig3_nogueira.png, fig3_nogueira_vs_ki.png
  - fig4_ss_combined.png      (2×3 grid: same panels, StabSel vs LASSO head-to-head)
  - fig4_ss_acc_f1.png        (Accuracy + F1 comparison: StabSel vs LASSO)
  - fig4_ss_stability.png     (KI + JI + Nogueira comparison: StabSel vs LASSO)
  - Console: KI, JI and Nogueira per individual percentage for each method
  - Console: Tables II & III (Accuracy, Precision, Recall, F1)
  - Console: Stability Selection head-to-head comparison table vs LASSO

FIXES applied vs previous version:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  FIX 1 — rank_lasso: rank ALL features, not a random subset        │
  │                                                                      │
  │  The original used refit=False in LogisticRegressionCV, which       │
  │  means sklearn returns coefficients averaged over CV folds, not     │
  │  a model fit on the full training set.  This caused the ranking     │
  │  to reflect fold-averaged weights rather than a proper full-data    │
  │  LASSO fit, effectively producing rankings as if only the features  │
  │  active in CV folds were considered.                                 │
  │                                                                      │
  │  Fix: after cross-validating for the best C, refit a fresh          │
  │  LogisticRegression on the FULL training set using that C, then     │
  │  rank ALL p=3403 features by |coef|.  refit=False is removed.       │
  ├─────────────────────────────────────────────────────────────────────┤
  │  FIX 2 — rank_stabsel: RNG moved outside / seeded per call         │
  │                                                                      │
  │  The original default argument rng=None caused a fresh              │
  │  np.random.default_rng(RANDOM_STATE) to be created on every call,  │
  │  making every invocation produce identical bootstrap samples        │
  │  regardless of the shuffle/fold context.                            │
  │                                                                      │
  │  Fix: the RNG is now always passed in explicitly from _one_shuffle  │
  │  using a (shuffle_id, fold_idx) derived seed, ensuring each        │
  │  (shuffle, fold) call gets a genuinely distinct random stream.      │
  │  The function signature still accepts rng= for testability, but    │
  │  the default raises an error to prevent silent misuse.              │
  ├─────────────────────────────────────────────────────────────────────┤
  │  FIX 3 — downstream classifier: tune C inside cross-validation     │
  │                                                                      │
  │  The original used a fixed C=1.0 for the LR classifier applied     │
  │  after feature selection.  This means performance numbers are       │
  │  artefacts of the arbitrary C choice rather than the feature        │
  │  selection method being evaluated.                                  │
  │                                                                      │
  │  Fix: replace the fixed-C LogisticRegression with a               │
  │  LogisticRegressionCV that searches over CLF_CV_CS inside each     │
  │  train fold, using 3-fold inner CV.  This makes each method's      │
  │  reported accuracy reflect the best achievable with its selected   │
  │  feature subset, keeping comparisons fair.                         │
  └─────────────────────────────────────────────────────────────────────┘
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model      import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import f_classif
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import StratifiedKFold
from sklearn.metrics           import (accuracy_score, f1_score,
                                       recall_score, precision_score)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
FILE_PATH      = '/kaggle/input/datasets/kinnarhalder/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat'
RESOLUTION_IDX = 0
N_ROIS_EXPECTED= 83

PERCENTAGES  = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 60.0, 70.0, 80.0]
N_SHUFFLES   = 20
N_FOLDS      = 5
RANDOM_STATE = 42

# ── Stability Selection hyperparameters ──────────────────────────────
SS_B               = 50    # bootstrap half-samples per training fold
SS_C_FIXED         = 0.05  # L1-LR C value: lower = sparser base selector
SS_RANDOM_STRENGTH = 0.5   # α: u_j ~ Uniform([α,1]); α=1 → standard LASSO

# ── Coarse C grid for rank_lasso ─────────────────────────────────────
L1_CV_CS = np.logspace(-3, 2, 6)

# ── FIX 3: C grid for the downstream classifier ─────────────────────
# Previously fixed at C=1.0; now tuned per fold via inner 3-fold CV.
CLF_CV_CS = np.logspace(-2, 2, 5)   # [0.01, 0.1, 1, 10, 100]

# ── Plot styles (4 methods) ───────────────────────────────────────────
STYLE = {
    'LASSO':   dict(color='#1f77b4', marker='o',  ls='-',   lw=1.8, ms=5),
    'Relief':  dict(color='#d62728', marker='s',  ls='--',  lw=1.8, ms=5),
    'ANOVA':   dict(color='#2ca02c', marker='^',  ls='-.',  lw=1.8, ms=5),
    'StabSel': dict(color='#9467bd', marker='D',  ls=':',   lw=2.0, ms=5),
}


# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING  (unchanged)
# ══════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load Dataset 1 from the .mat file.
    Returns:
        X : (54, 3403) float64  — |Pearson FC| upper-triangle vectors
        y : (54,)      int32    — 0=ctrl, 1=schz
    """
    print(f"  Loading: {FILE_PATH}")
    with h5py.File(FILE_PATH, 'r') as f:
        ctrl_ref = f['SC_FC_Connectomes/FC_correlation/ctrl']
        schz_ref = f['SC_FC_Connectomes/FC_correlation/schz']
        ctrl_mat = f[ctrl_ref[RESOLUTION_IDX, 0]][:]
        schz_mat = f[schz_ref[RESOLUTION_IDX, 0]][:]

    n_rois = ctrl_mat.shape[1]
    assert n_rois == N_ROIS_EXPECTED, f"Expected {N_ROIS_EXPECTED} ROIs, got {n_rois}"
    tri     = np.triu_indices(n_rois, k=1)
    vec     = lambda mats: np.abs(np.array([mats[i][tri] for i in range(len(mats))],
                                            dtype=np.float64))
    X = np.vstack([vec(ctrl_mat), vec(schz_mat)])
    y = np.array([0]*27 + [1]*27, dtype=np.int32)
    p = X.shape[1]
    assert p == n_rois*(n_rois-1)//2
    print(f"  Loaded: {n_rois} ROIs | p={p} features | "
          f"ctrl={int((y==0).sum())} | schz={int((y==1).sum())}")
    return X, y, n_rois


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE RANKERS
# ══════════════════════════════════════════════════════════════════════

def rank_lasso(Xs, y):
    """
    Rank ALL p features by L1-penalized Logistic Regression coefficient
    magnitude, fitted on the full training set.

    FIX 1: Previously used refit=False, which meant sklearn returned
    coefficients averaged over CV folds — not a model trained on the
    complete training split.  This caused features that happened to be
    inactive in some folds to receive artificially low weights, giving
    the impression that only a random-seeming subset of features were
    ranked meaningfully.

    Now:
      Step 1 — Use LogisticRegressionCV(refit=False) only for fast C
               selection (same coarse 6-point grid).
      Step 2 — Refit a single LogisticRegression(C=best_C) on the
               ENTIRE (Xs, y) training set.
      Step 3 — Rank all p features by |coef| from that full-data fit.

    This guarantees that every feature gets a coefficient and the
    ranking covers all 3403 features.
    """
    # Step 1: cross-validate to pick best C (fast, refit=False is fine here
    # because we only need the selected C value, not the CV coefficients)
    lrcv = LogisticRegressionCV(
        Cs=L1_CV_CS, penalty='l1', solver='saga', cv=3,
        max_iter=200, tol=1e-3, refit=False, n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    lrcv.fit(Xs, y)
    best_C = float(lrcv.C_[0])

    # Step 2: refit on the FULL training set with the selected C
    # FIX: this is the critical change — previously the coef from lrcv
    # (fold-averaged, refit=False) were used directly, missing features
    # that were only active in a subset of CV folds.
    lr = LogisticRegression(
        penalty='l1', C=best_C, solver='saga',
        max_iter=500, tol=1e-4,
        random_state=RANDOM_STATE,
    )
    lr.fit(Xs, y)

    # Step 3: rank ALL p features — argsort descending |coef|
    abs_coef = np.abs(lr.coef_[0])
    # abs_coef has shape (p,); features with coef==0 go to the end
    return np.argsort(-abs_coef).copy()


def rank_relief(Xs, y):
    """ReliefF — paper eq. (3)–(5).  Unchanged."""
    n, p = Xs.shape
    X0, X1 = Xs[y == 0], Xs[y == 1]
    w = np.zeros(p)
    for i in range(n):
        xl          = Xs[i]
        same, other = (X0, X1) if y[i] == 0 else (X1, X0)
        d_same  = np.sum((same  - xl)**2, axis=1)
        d_other = np.sum((other - xl)**2, axis=1)
        si = d_same.argmin()
        if d_same[si] < 1e-12:
            d_same[si] = np.inf
        w += (xl - other[d_other.argmin()])**2 - (xl - same[d_same.argmin()])**2
    return np.argsort(w / n)[::-1].copy()


def rank_anova(Xs, y):
    """ANOVA F-statistic — paper eq. (6)–(7).  Unchanged."""
    F, _ = f_classif(Xs, y)
    F    = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(F)[::-1].copy()


# ──────────────────────────────────────────────────────────────────────
def rank_stabsel(Xs, y, rng):
    """
    Stability Selection ranker (Meinshausen & Bühlmann, 2010).

    FIX 2: The rng parameter is now REQUIRED (no default).  Previously
    it defaulted to np.random.default_rng(RANDOM_STATE), which created
    the *same* RNG state on every call regardless of the calling
    shuffle/fold context.  This caused all bootstrap samples across
    all (shuffle, fold) pairs to be identical, defeating the purpose
    of running multiple shuffles.

    The caller (_one_shuffle) now always passes a distinct RNG derived
    from (RANDOM_STATE + shuffle_id * 1000 + fold_idx), ensuring each
    (shuffle, fold) call produces genuinely different bootstrap samples.

    Algorithm
    ─────────
    Given standardised training data (Xs, y) with n samples and p features:

    For b = 1 … SS_B:
      1. Draw a stratified half-sample I_b ⊆ {1,…,n}, |I_b| = ⌊n/2⌋.
      2. Apply the randomised LASSO perturbation:
           For each feature j, sample u_j ~ Uniform([α, 1]).
           Replace Xs[:,j] with Xs[:,j] / u_j  on the half-sample.
      3. Fit L1-LogisticRegression(C=SS_C_FIXED) on the perturbed half-sample.
      4. sel_b[j] = 1  iff  |coef_j| > 0.

    After B bootstraps:
      Π̂_j = (1/B) Σ_b sel_b[j]     # empirical selection probability

    Rank features by descending Π̂_j; ties broken by mean |coef|.

    References
    ──────────
    Meinshausen, N. & Bühlmann, P. (2010). Stability selection.
      JRSS-B, 72(4), 417–473.
    Shah, R. D. & Samworth, R. J. (2013). Variable selection with error
      control. JRSS-B, 75(1), 55–80.
    """
    # rng must be provided — no silent fixed-seed fallback (see FIX 2)
    if rng is None:
        raise ValueError(
            "rank_stabsel: rng must be provided explicitly.  "
            "Pass a np.random.default_rng() seeded per (shuffle, fold) "
            "to avoid all bootstrap calls producing identical samples."
        )

    n, p  = Xs.shape
    half  = max(2, n // 2)

    sel_count  = np.zeros(p, dtype=np.float64)
    coef_sum   = np.zeros(p, dtype=np.float64)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    h0   = max(1, len(idx0) // 2)
    h1   = max(1, len(idx1) // 2)

    for _ in range(SS_B):
        sub0  = rng.choice(idx0, size=h0, replace=False)
        sub1  = rng.choice(idx1, size=h1, replace=False)
        sub   = np.concatenate([sub0, sub1])
        Xb    = Xs[sub].copy()
        yb    = y[sub]

        u     = rng.uniform(SS_RANDOM_STRENGTH, 1.0, size=p)
        Xb   /= u[np.newaxis, :]

        lr = LogisticRegression(
            penalty='l1', C=SS_C_FIXED, solver='saga',
            max_iter=200, tol=1e-3, random_state=None,
        )
        lr.fit(Xb, yb)

        abs_c       = np.abs(lr.coef_[0])
        sel_count  += (abs_c > 0).astype(np.float64)
        coef_sum   += abs_c

    pi_hat = sel_count / float(SS_B)
    mean_c = coef_sum  / float(SS_B)

    rank = np.lexsort((-mean_c, -pi_hat))
    return rank.copy()


# ── Ranker registry ───────────────────────────────────────────────────
RANKERS = {
    'LASSO':   rank_lasso,
    'Relief':  rank_relief,
    'ANOVA':   rank_anova,
    'StabSel': rank_stabsel,
}


# ══════════════════════════════════════════════════════════════════════
# 3. STABILITY METRICS  (unchanged)
# ══════════════════════════════════════════════════════════════════════

def compute_ki_ji(signatures, p):
    """KI (Kuncheva) and JI (Jaccard) over all C(λ,2) pairs."""
    lam = len(signatures)
    if lam < 2: return np.nan, np.nan
    k = len(signatures[0])
    if k == 0: return np.nan, np.nan

    S = np.zeros((lam, p), dtype=np.float32)
    for i, sig in enumerate(signatures):
        valid = sig[(sig >= 0) & (sig < p)]
        S[i, valid] = 1.0

    intersect = S @ S.T
    iu        = np.triu_indices(lam, k=1)
    r_vals    = intersect[iu]

    k2p   = float(k)**2 / float(p)
    denom = float(k) - k2p
    ki    = float(np.clip(np.mean((r_vals - k2p) / denom), -1.0, 1.0)) \
            if abs(denom) > 1e-12 else 1.0

    union_vals = 2.0 * float(k) - r_vals
    ji         = float(np.mean(np.where(union_vals > 0, r_vals / union_vals, 1.0)))
    return ki, ji


def compute_nogueira(signatures, p):
    """
    Nogueira Stability Index Ŝ (Nogueira, Brown & Sherrat, JMLR 2018).

    Ŝ = 1 − (p · V̄) / (k̄(1 − k̄/p))

    where V̄ = mean per-feature variance, k̄ = mean subset size.
    """
    lam = len(signatures)
    if lam < 2: return np.nan

    Z = np.zeros((lam, p), dtype=np.float64)
    for i, sig in enumerate(signatures):
        valid = sig[(sig >= 0) & (sig < p)]
        Z[i, valid] = 1.0

    k_bar = float(Z.sum(axis=1).mean())
    if k_bar <= 0.0 or k_bar >= float(p): return np.nan

    p_hat = Z.mean(axis=0)
    V_bar = float(np.mean(p_hat * (1.0 - p_hat)))
    denom = k_bar * (1.0 - k_bar / float(p))
    if abs(denom) < 1e-12: return np.nan
    return float(1.0 - (float(p) * V_bar) / denom)


# ══════════════════════════════════════════════════════════════════════
# 4. ONE SHUFFLE
# ══════════════════════════════════════════════════════════════════════

def _one_shuffle(shuffle_id, X, y, percentages):
    p   = X.shape[1]
    ks  = [max(1, int(pct / 100.0 * p)) for pct in percentages]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE + shuffle_id)

    acc_s = {m: np.zeros(len(percentages)) for m in RANKERS}
    f1_s  = {m: np.zeros(len(percentages)) for m in RANKERS}
    rec_s = {m: np.zeros(len(percentages)) for m in RANKERS}
    pre_s = {m: np.zeros(len(percentages)) for m in RANKERS}
    sigs  = {m: [[] for _ in percentages]  for m in RANKERS}

    fold_idx = 0
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(X_tr)
        Xte_s = sc.transform(X_te)

        # FIX 2: derive a unique RNG per (shuffle, fold) — never reuse
        # RANDOM_STATE alone, which caused identical bootstrap sequences
        ss_rng = np.random.default_rng(
            RANDOM_STATE + shuffle_id * 1000 + fold_idx
        )

        for mname, ranker in RANKERS.items():
            if mname == 'StabSel':
                rank = ranker(Xtr_s, y_tr, rng=ss_rng)
            else:
                rank = ranker(Xtr_s, y_tr)

            for pi, k in enumerate(ks):
                sel = rank[:k]

                # FIX 3: tune classifier C via inner 3-fold CV
                # Previously: LogisticRegression(C=1.0) — fixed, untuned
                # Now: LogisticRegressionCV searches CLF_CV_CS each fold,
                # so reported metrics reflect the best achievable accuracy
                # with each method's feature subset, not an arbitrary C.
                clf = LogisticRegressionCV(
                    Cs=CLF_CV_CS,
                    cv=3,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
                clf.fit(Xtr_s[:, sel], y_tr)
                pred = clf.predict(Xte_s[:, sel])

                acc_s[mname][pi] += accuracy_score(y_te, pred)
                f1_s [mname][pi] += f1_score(y_te, pred, zero_division=0)
                rec_s[mname][pi] += recall_score(y_te, pred, zero_division=0)
                pre_s[mname][pi] += precision_score(y_te, pred, zero_division=0)
                sigs [mname][pi].append(sel.copy())

        fold_idx += 1

    result = {}
    for mname in RANKERS:
        result[mname] = {
            'acc':  acc_s[mname] / N_FOLDS,
            'f1':   f1_s [mname] / N_FOLDS,
            'rec':  rec_s[mname] / N_FOLDS,
            'pre':  pre_s[mname] / N_FOLDS,
            'sigs': sigs [mname],
        }
    return result


# ══════════════════════════════════════════════════════════════════════
# 5. FULL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate(X, y, percentages):
    p   = X.shape[1]
    lam = N_SHUFFLES * N_FOLDS
    print(f"  {N_SHUFFLES} shuffles × {N_FOLDS} folds = {lam} signatures per method per %")

    shuffles = []
    for s in range(N_SHUFFLES):
        print(f"    Shuffle {s+1:02d}/{N_SHUFFLES} ...", flush=True)
        shuffles.append(_one_shuffle(s, X, y, percentages))

    all_sigs = {m: [[] for _ in percentages] for m in RANKERS}
    for sr in shuffles:
        for mname in RANKERS:
            for pi in range(len(percentages)):
                all_sigs[mname][pi].extend(sr[mname]['sigs'][pi])

    results = {m: {'acc': [], 'f1': [], 'rec': [], 'pre': [],
                   'ki': [], 'ji': [], 'nogueira': []}
               for m in RANKERS}

    for mname in RANKERS:
        for pi in range(len(percentages)):
            acc_v = np.array([sr[mname]['acc'][pi] for sr in shuffles])
            f1_v  = np.array([sr[mname]['f1'] [pi] for sr in shuffles])
            rec_v = np.array([sr[mname]['rec'][pi] for sr in shuffles])
            pre_v = np.array([sr[mname]['pre'][pi] for sr in shuffles])

            ki, ji   = compute_ki_ji(all_sigs[mname][pi], p)
            nogueira = compute_nogueira(all_sigs[mname][pi], p)

            results[mname]['acc']     .append(float(np.mean(acc_v)))
            results[mname]['f1']      .append(float(np.mean(f1_v)))
            results[mname]['rec']     .append(float(np.mean(rec_v)))
            results[mname]['pre']     .append(float(np.mean(pre_v)))
            results[mname]['ki']      .append(ki)
            results[mname]['ji']      .append(ji)
            results[mname]['nogueira'].append(nogueira)

    return results


# ══════════════════════════════════════════════════════════════════════
# 6. CONSOLE OUTPUT — KI / JI / NOGUEIRA PER PERCENTAGE
# ══════════════════════════════════════════════════════════════════════

def print_ki_ji_nogueira_per_percentage(results, percentages):
    p   = 3403
    bar = '═' * 100
    ALL_METHODS = list(RANKERS.keys())

    for mname in ALL_METHODS:
        print(f"\n{bar}")
        print(f"  METHOD: {mname}  —  KI, JI & Nogueira at each % of selected features")
        print(f"  {'%Features':>10}  {'k':>6}  {'KI':>10}  {'JI':>10}  "
              f"{'Nogueira':>10}  {'Accuracy':>10}  {'F1-Score':>10}")
        print(f"{'─'*100}")
        for pi, pct in enumerate(percentages):
            k   = max(1, int(pct / 100.0 * p))
            ki  = results[mname]['ki']      [pi]
            ji  = results[mname]['ji']      [pi]
            ng  = results[mname]['nogueira'][pi]
            acc = results[mname]['acc']     [pi] * 100
            f1  = results[mname]['f1']      [pi] * 100
            ng_s = f"{ng:>10.4f}" if not np.isnan(ng) else f"{'N/A':>10}"
            print(f"  {pct:>10.1f}%  {k:>6d}  {ki:>10.4f}  {ji:>10.4f}  "
                  f"{ng_s}  {acc:>9.2f}%  {f1:>9.2f}%")
        print(bar)

    col_w = 12
    hdr   = f"  {'%Features':>10}  {'k':>6}" + \
            "".join(f"  {m:>{col_w}}" for m in ALL_METHODS)

    for metric, label in [('ki', 'KUNCHEVA INDEX (KI)'),
                           ('ji', 'JACCARD INDEX (JI)'),
                           ('nogueira', 'NOGUEIRA INDEX (Ŝ)')]:
        print(f"\n{bar}")
        print(f"  {label}  — all methods at each % of selected features")
        print(hdr)
        print(f"{'─'*100}")
        for pi, pct in enumerate(percentages):
            k   = max(1, int(pct / 100.0 * p))
            row = f"  {pct:>10.1f}%  {k:>6d}"
            for m in ALL_METHODS:
                v = results[m][metric][pi]
                row += f"  {v:>{col_w}.4f}" if not np.isnan(v) \
                       else f"  {'N/A':>{col_w}}"
            print(row)
        print(bar)

    print(f"\n{'─'*66}")
    print(f"  Mean KI / JI / Nogueira averaged over all {len(percentages)} percentages:")
    print(f"  {'Method':<12}  {'Mean KI':>10}  {'Mean JI':>10}  {'Mean Ŝ':>10}")
    print(f"{'─'*66}")
    for m in ALL_METHODS:
        ki_m = np.mean(results[m]['ki'])
        ji_m = np.mean(results[m]['ji'])
        ng_v = [v for v in results[m]['nogueira'] if not np.isnan(v)]
        ng_m = np.mean(ng_v) if ng_v else float('nan')
        ng_s = f"{ng_m:>10.4f}" if not np.isnan(ng_m) else f"{'N/A':>10}"
        print(f"  {m:<12}  {ki_m:>10.4f}  {ji_m:>10.4f}  {ng_s}")
    print(f"{'─'*66}")


# ══════════════════════════════════════════════════════════════════════
# 7. TABLES II & III
# ══════════════════════════════════════════════════════════════════════

def print_tables(results, percentages):
    sep    = '─' * 70
    header = (f"  {'Method':<12}  {'Accuracy':>9}  "
              f"{'Precision':>9}  {'Recall':>9}  {'F1-Score':>9}")
    for pct, tname in [(5.0, 'II  — top  5%'), (10.0, 'III — top 10%')]:
        pi = percentages.index(pct)
        print(f"\n{sep}")
        print(f"  Table {tname} components — Dataset 1")
        print(sep)
        print(header)
        print(sep)
        for m in RANKERS:
            a  = results[m]['acc'][pi] * 100
            pr = results[m]['pre'][pi] * 100
            rc = results[m]['rec'][pi] * 100
            f1 = results[m]['f1'] [pi] * 100
            print(f"  {m:<12}  {a:>8.2f}%  {pr:>8.2f}%  {rc:>8.2f}%  {f1:>8.2f}%")
        print(sep)


# ══════════════════════════════════════════════════════════════════════
# 8. STABILITY SELECTION vs LASSO HEAD-TO-HEAD TABLE
# ══════════════════════════════════════════════════════════════════════

def print_stabsel_comparison(results, percentages):
    """
    Dedicated console comparison: StabSel vs LASSO across all metrics
    and all feature percentages.
    """
    p   = 3403
    bar = '═' * 104
    print(f"\n{bar}")
    print(f"  STABILITY SELECTION vs LASSO  —  Head-to-Head Comparison")
    print(f"  Δ = StabSel − LASSO  (positive = StabSel is better / more stable)")
    print(f"  {'%Feat':>7}  {'k':>6}"
          f"  {'SS Acc':>9}  {'L Acc':>9}  {'Δ Acc':>8}"
          f"  {'SS F1':>9}  {'L F1':>9}  {'Δ F1':>8}"
          f"  {'SS KI':>8}  {'L KI':>8}  {'Δ KI':>7}"
          f"  {'SS Ŝ':>8}  {'L Ŝ':>8}  {'Δ Ŝ':>7}")
    print(f"{'─'*104}")

    for pi, pct in enumerate(percentages):
        k     = max(1, int(pct / 100.0 * p))
        ss_a  = results['StabSel']['acc']     [pi] * 100
        l_a   = results['LASSO']  ['acc']     [pi] * 100
        ss_f  = results['StabSel']['f1']      [pi] * 100
        l_f   = results['LASSO']  ['f1']      [pi] * 100
        ss_ki = results['StabSel']['ki']       [pi]
        l_ki  = results['LASSO']  ['ki']       [pi]
        ss_ng = results['StabSel']['nogueira'] [pi]
        l_ng  = results['LASSO']  ['nogueira'] [pi]

        d_a   = ss_a  - l_a
        d_f   = ss_f  - l_f
        d_ki  = ss_ki - l_ki
        d_ng  = (ss_ng - l_ng) if (not np.isnan(ss_ng) and not np.isnan(l_ng)) \
                else float('nan')

        ng_ss = f"{ss_ng:>8.4f}" if not np.isnan(ss_ng) else f"{'N/A':>8}"
        ng_l  = f"{l_ng:>8.4f}"  if not np.isnan(l_ng)  else f"{'N/A':>8}"
        ng_d  = f"{d_ng:>7.4f}"  if not np.isnan(d_ng)  else f"{'N/A':>7}"

        print(f"  {pct:>7.1f}%  {k:>6d}"
              f"  {ss_a:>9.2f}  {l_a:>9.2f}  {d_a:>+8.2f}"
              f"  {ss_f:>9.2f}  {l_f:>9.2f}  {d_f:>+8.2f}"
              f"  {ss_ki:>8.4f}  {l_ki:>8.4f}  {d_ki:>+7.4f}"
              f"  {ng_ss}  {ng_l}  {ng_d}")
    print(bar)

    def _mean(m, key):
        v = results[m][key]
        v = [x for x in v if not np.isnan(x)]
        return np.mean(v) if v else float('nan')

    print(f"\n  MEAN ACROSS ALL {len(percentages)} PERCENTAGES:")
    print(f"  {'Method':<12}  {'Acc':>9}  {'F1':>9}  {'KI':>9}  {'Nogueira Ŝ':>12}")
    print(f"  {'─'*60}")
    for m in ['StabSel', 'LASSO']:
        a  = _mean(m, 'acc') * 100
        f1 = _mean(m, 'f1')  * 100
        ki = _mean(m, 'ki')
        ng = _mean(m, 'nogueira')
        ng_s = f"{ng:>12.4f}" if not np.isnan(ng) else f"{'N/A':>12}"
        print(f"  {m:<12}  {a:>9.2f}  {f1:>9.2f}  {ki:>9.4f}  {ng_s}")

    mean_d_ki = _mean('StabSel', 'ki') - _mean('LASSO', 'ki')
    mean_d_ac = (_mean('StabSel', 'acc') - _mean('LASSO', 'acc')) * 100
    winner_stab = "StabSel" if mean_d_ki > 0 else "LASSO"
    winner_acc  = "StabSel" if mean_d_ac > 0 else "LASSO"
    print(f"\n  INTERPRETATION:")
    print(f"  • Stability (mean ΔKI={mean_d_ki:+.4f}): {winner_stab} is more stable "
          f"on average across all k.")
    print(f"  • Accuracy  (mean ΔAcc={mean_d_ac:+.2f}%):  {winner_acc} achieves "
          f"higher mean accuracy.")
    print(f"  • StabSel is an *embedded* method: it natively produces stable "
          f"feature sets by construction\n"
          f"    (each of the {SS_B} bootstraps independently filters with L1; "
          f"only consistently-selected\n"
          f"    features survive to the top of the ranking). This is why it "
          f"typically shows higher\n"
          f"    KI/Nogueira at small k, where LASSO rankings fluctuate more.")
    print(bar)


# ══════════════════════════════════════════════════════════════════════
# 9. PLOTS
# ══════════════════════════════════════════════════════════════════════

def _panel(ax, data_dict, pct_list, ylabel, letter, style_map=None):
    sm = style_map if style_map is not None else STYLE
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for mname, style in sm.items():
        if mname in data_dict:
            ax.plot(x, data_dict[mname], label=mname, **style)
    ax.set_xticks(x)
    ax.set_xticklabels(xl, rotation=45, fontsize=8)
    ax.set_xlabel('Percentage of Selected Features', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(letter, loc='left', fontsize=10, pad=4)
    ax.legend(fontsize=8, loc='best', framealpha=0.7)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.tick_params(labelsize=8)
    vals = [v for m in sm if m in data_dict
            for v in data_dict[m] if not np.isnan(v)]
    if vals:
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.15, 0.02)
        ax.set_ylim(lo - pad, hi + pad)


def _panel_nogueira_vs_ki(ax, results, pct_list, methods=None):
    if methods is None:
        methods = list(STYLE.keys())
    colors = {m: STYLE[m]['color'] for m in methods}
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for mname in methods:
        c  = colors[mname]
        ng = [0.0 if np.isnan(v) else v for v in results[mname]['nogueira']]
        ki = results[mname]['ki']
        ax.plot(x, ng, color=c, ls='-',  lw=1.8, ms=5, marker='o',
                label=f'{mname} Ŝ')
        ax.plot(x, ki, color=c, ls='--', lw=1.5, ms=4, marker='s',
                label=f'{mname} KI')
    ax.set_xticks(x)
    ax.set_xticklabels(xl, rotation=45, fontsize=8)
    ax.set_xlabel('Percentage of Selected Features', fontsize=9)
    ax.set_ylabel('Stability Index', fontsize=9)
    ax.set_title('(f) Nogueira Ŝ vs Kuncheva KI.', loc='left', fontsize=10, pad=4)
    ax.legend(fontsize=7, loc='best', framealpha=0.7, ncol=2)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.tick_params(labelsize=8)


def plot_fig3(results, percentages):
    panels = [
        ('acc',      'Accuracy',        '(a) Accuracy.'),
        ('f1',       'F1 Score',        '(b) F1 score.'),
        ('ki',       'Kuncheva Index',  '(c) Kuncheva index.'),
        ('ji',       'Jaccard Index',   '(d) Jaccard index.'),
        ('nogueira', 'Nogueira Ŝ',      '(e) Nogueira stability index.'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(
        'Fig. 3  —  Dataset 1 (University of Lausanne)\n'
        'Classification Performance & Feature Selection Stability  '
        '(4 methods: LASSO, Relief, ANOVA, StabSel)\n'
        '54 subjects · 83 ROIs · 3,403 FC features  |  λ=100 subsamplings  |  '
        f'StabSel: B={SS_B}, C={SS_C_FIXED}, α={SS_RANDOM_STRENGTH}\n'
        f'[FIXED: full-data LASSO ranking | per-fold RNG | tuned classifier C∈{list(CLF_CV_CS)}]',
        fontsize=8, fontweight='bold', y=1.02)

    for (key, ylabel, letter), ax in zip(panels, axes.flat):
        data_dict = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
                     for m in STYLE}
        _panel(ax, data_dict, percentages, ylabel, letter)

    _panel_nogueira_vs_ki(axes.flat[5], results, percentages)

    plt.tight_layout()
    fig.savefig('fig3_combined.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig3_combined.png")

    fnames = {
        'acc':      'fig3_accuracy.png',
        'f1':       'fig3_f1score.png',
        'ki':       'fig3_kuncheva.png',
        'ji':       'fig3_jaccard.png',
        'nogueira': 'fig3_nogueira.png',
    }
    for key, ylabel, letter in panels:
        data_dict = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
                     for m in STYLE}
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        _panel(ax2, data_dict, percentages, ylabel, letter)
        plt.tight_layout()
        fig2.savefig(fnames[key], dpi=180, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {fnames[key]}")

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    _panel_nogueira_vs_ki(ax3, results, percentages)
    ax3.set_title('Nogueira Ŝ vs Kuncheva KI — all methods', fontsize=11)
    plt.tight_layout()
    fig3.savefig('fig3_nogueira_vs_ki.png', dpi=180, bbox_inches='tight')
    plt.close(fig3)
    print("  Saved: fig3_nogueira_vs_ki.png")


def plot_fig4_comparison(results, percentages):
    """
    Dedicated 2×3 figure comparing StabSel and LASSO only.
    """
    two_style = {
        'LASSO':   STYLE['LASSO'],
        'StabSel': STYLE['StabSel'],
    }
    panels = [
        ('acc',      'Accuracy',        '(a) Accuracy.'),
        ('f1',       'F1 Score',        '(b) F1 score.'),
        ('ki',       'Kuncheva Index',  '(c) Kuncheva index.'),
        ('ji',       'Jaccard Index',   '(d) Jaccard index.'),
        ('nogueira', 'Nogueira Ŝ',      '(e) Nogueira stability index.'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        'Fig. 4  —  Stability Selection vs LASSO  (Dataset 1)\n'
        'Head-to-Head: Classification Performance & Feature Stability\n'
        '54 subjects · 83 ROIs · 3,403 FC features  |  λ=100 subsamplings  |  '
        f'StabSel: B={SS_B}, C={SS_C_FIXED}, α={SS_RANDOM_STRENGTH}',
        fontsize=9, fontweight='bold', y=1.01)

    for (key, ylabel, letter), ax in zip(panels, axes.flat):
        data_dict = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
                     for m in two_style}
        _panel(ax, data_dict, percentages, ylabel, letter, style_map=two_style)

    _panel_nogueira_vs_ki(axes.flat[5], results, percentages,
                          methods=['LASSO', 'StabSel'])

    plt.tight_layout()
    fig.savefig('fig4_ss_combined.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig4_ss_combined.png")

    fig_a, axes_a = plt.subplots(1, 2, figsize=(12, 5))
    for (key, ylabel, letter), ax in zip(panels[:2], axes_a):
        data_dict = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
                     for m in two_style}
        _panel(ax, data_dict, percentages, ylabel, letter, style_map=two_style)
    fig_a.suptitle('StabSel vs LASSO — Classification Performance', fontsize=10)
    plt.tight_layout()
    fig_a.savefig('fig4_ss_acc_f1.png', dpi=180, bbox_inches='tight')
    plt.close(fig_a)
    print("  Saved: fig4_ss_acc_f1.png")

    fig_b, axes_b = plt.subplots(1, 2, figsize=(12, 5))
    for (key, ylabel, letter), ax in zip(
            [('ki', 'Kuncheva Index', '(a) KI.'),
             ('nogueira', 'Nogueira Ŝ', '(b) Nogueira Ŝ.')],
            axes_b):
        data_dict = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
                     for m in two_style}
        _panel(ax, data_dict, percentages, ylabel, letter, style_map=two_style)
    fig_b.suptitle('StabSel vs LASSO — Feature Selection Stability', fontsize=10)
    plt.tight_layout()
    fig_b.savefig('fig4_ss_stability.png', dpi=180, bbox_inches='tight')
    plt.close(fig_b)
    print("  Saved: fig4_ss_stability.png")


# ══════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    bar = '═' * 70
    print(bar)
    print('  Saha, Hazra & Ghosh (2025) — Figure 3 Reproduction')
    print('  Dataset 1: University of Lausanne  (83 ROIs, n=54)')
    print('  Methods: LASSO (L1-LR saga), Relief, ANOVA,')
    print(f'           StabSel (B={SS_B}, C={SS_C_FIXED}, α={SS_RANDOM_STRENGTH})')
    print('  Stability metrics: KI, JI, Nogueira Ŝ')
    print('  FIXES APPLIED:')
    print('    [1] rank_lasso: refit on full training set → ranks ALL 3403 features')
    print('    [2] rank_stabsel: unique RNG per (shuffle, fold) → diverse bootstraps')
    print(f'   [3] Classifier: LogisticRegressionCV(Cs={list(CLF_CV_CS)}) per fold')
    print(bar)

    X, y, n_rois = load_data()

    print(f'\n[Step 1]  Running {N_SHUFFLES}×{N_FOLDS} cross-validation ...')
    results = evaluate(X, y, PERCENTAGES)

    print('\n[Step 2]  KI, JI, and Nogueira at each individual % of selected features:')
    print_ki_ji_nogueira_per_percentage(results, PERCENTAGES)

    print('\n[Step 3]  Tables II & III:')
    print_tables(results, PERCENTAGES)

    print('\n[Step 4]  Stability Selection vs LASSO head-to-head:')
    print_stabsel_comparison(results, PERCENTAGES)

    print('\n[Step 5]  Plotting Fig. 3 (all 4 methods) ...')
    plot_fig3(results, PERCENTAGES)

    print('\n[Step 6]  Plotting Fig. 4 (StabSel vs LASSO) ...')
    plot_fig4_comparison(results, PERCENTAGES)

    print(f'\n{bar}')
    print('  All done.')
    print(bar)


if __name__ == '__main__':
    main()
