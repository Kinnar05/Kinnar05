"""
Extended Reproduction of Saha, Hazra & Ghosh (2025) + Bai et al. (NeurIPS 2019)
================================================================================
Evaluates ALL 10 feature selection methods for Accuracy, F1, KI, JI & Nogueira:

  ┌──── From the original code (Saha et al. 2025) ────────────────────────────┐
  │  1. LASSO       — L1-penalised Logistic Regression (saga)                 │
  │  2. Relief      — ReliefF nearest-hit / nearest-miss weighting            │
  │  3. ANOVA       — ANOVA F-statistic univariate filter                     │
  │  4. StabSel     — Meinshausen & Bühlmann Stability Selection              │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌──── From Bai et al. NeurIPS 2019 (Table 2 / Table 3) ─────────────────────┐
  │  5. ULasso      — Uncorrelated Lasso (Chen et al. 2013) approx.           │
  │  6. FusedLasso  — Fused Lasso (Tibshirani et al. 2005) approx.            │
  │  7. GroupLasso  — Group Lasso (block-soft-threshold, equal groups)        │
  │  8. InLasso     — Interacted Lasso (Zhang et al. 2017) approx.            │
  │  9. InFusedLasso— Structural Interacting Fused Lasso (Bai et al. 2019)    │
  │ 10. InElasticNet— Interacted Elastic Net (Cui et al. 2019) approx.        │
  └────────────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION NOTES FOR METHODS 5–10
──────────────────────────────────────
The Bai et al. methods operate on the *feature interaction matrix* U (Eq. 4 in
the paper) built from kernel-based graph representations.  Computing the exact
U for all C(3403,2) ≈ 5.8 million feature pairs on 54 samples is prohibitively
expensive (and the paper uses dedicated Matlab code).  We instead implement
practical *approximations* that preserve the defining characteristic of each
method while remaining tractable:

  • ULasso      : L1-LR with an additional uncorrelation penalty term.
                  Approximated by iteratively down-weighting features that are
                  highly correlated with already-selected features (greedy
                  uncorrelated selection).

  • FusedLasso  : L1 + fused penalty (successive coefficient differences).
                  Approximated by sorting features and applying a 1-D signal-
                  smoothness prior via a difference-penalty on ranked coefficients
                  (proximal gradient on sorted abs-correlation).

  • GroupLasso  : Groups of equal size; block-soft-threshold ranking.
                  Features ranked by group-level L2 norm; within each group,
                  ranked by individual correlation magnitude.

  • InLasso     : Interacted Lasso — augments L1-LR with pairwise interaction
                  scores (diagonal of X^T·Sigma·X, where Sigma is the class
                  covariance).  Features ranked by interaction-adjusted weights.

  • InFusedLasso: Structural Interacting Fused Lasso — combines InLasso
                  interaction scores with a fused-lasso successive-difference
                  penalty.  Features ranked by interaction-fused score.

  • InElasticNet: Interacted Elastic Net — L1+L2 penalised LR with interaction
                  score reweighting (L2 term stabilises in high-p setting).

All approximations rank features BEFORE classification; the downstream
classifier (tuned LogisticRegressionCV) is identical across all methods,
ensuring fair comparisons.

FIXES (inherited from original code)
──────────────────────────────────────
  [1] LASSO: refit on full training set — ranks ALL 3403 features
  [2] StabSel: unique RNG per (shuffle, fold) — diverse bootstraps
  [3] Classifier: LogisticRegressionCV(Cs=CLF_CV_CS) per fold
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations

from sklearn.linear_model      import LogisticRegression, LogisticRegressionCV, Ridge
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
SS_B               = 50
SS_C_FIXED         = 0.05
SS_RANDOM_STRENGTH = 0.5

# ── GroupLasso group size ─────────────────────────────────────────────
GL_GROUP_SIZE = 50   # group every 50 consecutive features

# ── Coarse C grids ───────────────────────────────────────────────────
L1_CV_CS  = np.logspace(-3, 2, 6)
CLF_CV_CS = np.logspace(-2, 2, 5)    # [0.01, 0.1, 1, 10, 100]

# ── Plot colour / marker palette — 10 methods ────────────────────────
STYLE = {
    'LASSO':        dict(color='#1f77b4', marker='o',  ls='-',    lw=1.8, ms=5),
    'Relief':       dict(color='#d62728', marker='s',  ls='--',   lw=1.8, ms=5),
    'ANOVA':        dict(color='#2ca02c', marker='^',  ls='-.',   lw=1.8, ms=5),
    'StabSel':      dict(color='#9467bd', marker='D',  ls=':',    lw=2.0, ms=5),
    'ULasso':       dict(color='#ff7f0e', marker='v',  ls='-',    lw=1.8, ms=5),
    'FusedLasso':   dict(color='#8c564b', marker='P',  ls='--',   lw=1.8, ms=5),
    'GroupLasso':   dict(color='#e377c2', marker='X',  ls='-.',   lw=1.8, ms=5),
    'InLasso':      dict(color='#17becf', marker='*',  ls=':',    lw=2.0, ms=7),
    'InFusedLasso': dict(color='#bcbd22', marker='h',  ls='-',    lw=2.0, ms=6),
    'InElasticNet': dict(color='#7f7f7f', marker='d',  ls='--',   lw=1.8, ms=5),
}


# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_data():
    print(f"  Loading: {FILE_PATH}")
    with h5py.File(FILE_PATH, 'r') as f:
        ctrl_ref = f['SC_FC_Connectomes/FC_correlation/ctrl']
        schz_ref = f['SC_FC_Connectomes/FC_correlation/schz']
        ctrl_mat = f[ctrl_ref[RESOLUTION_IDX, 0]][:]
        schz_mat = f[schz_ref[RESOLUTION_IDX, 0]][:]

    n_rois = ctrl_mat.shape[1]
    assert n_rois == N_ROIS_EXPECTED, f"Expected {N_ROIS_EXPECTED} ROIs, got {n_rois}"
    tri = np.triu_indices(n_rois, k=1)
    vec = lambda mats: np.abs(np.array([mats[i][tri] for i in range(len(mats))],
                                        dtype=np.float64))
    X = np.vstack([vec(ctrl_mat), vec(schz_mat)])
    y = np.array([0]*27 + [1]*27, dtype=np.int32)
    p = X.shape[1]
    assert p == n_rois*(n_rois-1)//2
    print(f"  Loaded: {n_rois} ROIs | p={p} features | "
          f"ctrl={int((y==0).sum())} | schz={int((y==1).sum())}")
    return X, y, n_rois


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE RANKERS — original 4
# ══════════════════════════════════════════════════════════════════════

def rank_lasso(Xs, y, **_):
    """L1-LR, refit on full training set (FIX 1)."""
    lrcv = LogisticRegressionCV(
        Cs=L1_CV_CS, penalty='l1', solver='saga', cv=3,
        max_iter=200, tol=1e-3, refit=False, n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    lrcv.fit(Xs, y)
    best_C = float(lrcv.C_[0])
    lr = LogisticRegression(
        penalty='l1', C=best_C, solver='saga',
        max_iter=500, tol=1e-4, random_state=RANDOM_STATE,
    )
    lr.fit(Xs, y)
    return np.argsort(-np.abs(lr.coef_[0])).copy()


def rank_relief(Xs, y, **_):
    """ReliefF nearest-hit / nearest-miss."""
    n, p = Xs.shape
    X0, X1 = Xs[y == 0], Xs[y == 1]
    w = np.zeros(p)
    for i in range(n):
        xl = Xs[i]
        same, other = (X0, X1) if y[i] == 0 else (X1, X0)
        d_same  = np.sum((same  - xl)**2, axis=1)
        d_other = np.sum((other - xl)**2, axis=1)
        si = d_same.argmin()
        if d_same[si] < 1e-12:
            d_same[si] = np.inf
        w += (xl - other[d_other.argmin()])**2 - (xl - same[d_same.argmin()])**2
    return np.argsort(w / n)[::-1].copy()


def rank_anova(Xs, y, **_):
    """ANOVA F-statistic."""
    F, _ = f_classif(Xs, y)
    F    = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(F)[::-1].copy()


def rank_stabsel(Xs, y, rng=None, **_):
    """
    Stability Selection (Meinshausen & Bühlmann 2010).
    rng MUST be provided — unique per (shuffle, fold) (FIX 2).
    """
    if rng is None:
        raise ValueError("rank_stabsel: rng must be provided explicitly.")
    n, p  = Xs.shape
    idx0  = np.where(y == 0)[0]
    idx1  = np.where(y == 1)[0]
    h0, h1 = max(1, len(idx0)//2), max(1, len(idx1)//2)

    sel_count = np.zeros(p, dtype=np.float64)
    coef_sum  = np.zeros(p, dtype=np.float64)

    for _ in range(SS_B):
        sub  = np.concatenate([
            rng.choice(idx0, size=h0, replace=False),
            rng.choice(idx1, size=h1, replace=False),
        ])
        Xb   = Xs[sub].copy()
        yb   = y[sub]
        u    = rng.uniform(SS_RANDOM_STRENGTH, 1.0, size=p)
        Xb  /= u[np.newaxis, :]
        lr   = LogisticRegression(
            penalty='l1', C=SS_C_FIXED, solver='saga',
            max_iter=200, tol=1e-3, random_state=None,
        )
        lr.fit(Xb, yb)
        abs_c       = np.abs(lr.coef_[0])
        sel_count  += (abs_c > 0).astype(np.float64)
        coef_sum   += abs_c

    pi_hat = sel_count / float(SS_B)
    mean_c = coef_sum  / float(SS_B)
    order  = np.argsort(-pi_hat, kind='stable')
    return order[np.argsort(-mean_c[order], kind='stable')].copy()


# ══════════════════════════════════════════════════════════════════════
# 3. FEATURE RANKERS — methods 5–10 (Bai et al. family)
# ══════════════════════════════════════════════════════════════════════

def _abs_corr_with_y(Xs, y):
    """Absolute Pearson correlation of each feature with y (class label)."""
    y_c  = y.astype(np.float64) - y.mean()
    Xc   = Xs - Xs.mean(axis=0)
    cov  = Xc.T @ y_c
    std_X = np.sqrt((Xc**2).sum(axis=0)) + 1e-12
    std_y = np.sqrt((y_c**2).sum()) + 1e-12
    return np.abs(cov / (std_X * std_y))


def rank_ulasso(Xs, y, **_):
    """
    ULasso — Uncorrelated Lasso (Chen et al. 2013).
    Greedy selection that penalises features redundant to already-chosen ones.
    Score(j) = corr(f_j, y) / (1 + max_{k in S} |corr(f_j, f_k)|)
    iteratively updated as the selected set S grows.
    We rank ALL p features by their final uncorrelated score.
    """
    n, p   = Xs.shape
    corr_y = _abs_corr_with_y(Xs, y)          # (p,)

    # Feature–feature abs-correlation matrix (memory-efficient: compute on-demand)
    # For p=3403 the full matrix is ~87 MB — manageable
    Xc     = Xs - Xs.mean(axis=0)
    std_X  = np.sqrt((Xc**2).sum(axis=0)) + 1e-12
    Xn     = Xc / std_X                       # column-normalised

    scores    = corr_y.copy()
    selected  = []
    rank_list = []

    remaining = list(range(p))
    max_corr_arr = np.zeros(p, dtype=np.float64)   # running max-corr to selected

    for _ in range(p):
        if not remaining:
            break
        rem = np.array(remaining, dtype=np.intp)
        best_local = int(np.argmax(scores[rem]))
        best_idx   = int(rem[best_local])
        rank_list.append(best_idx)
        selected.append(best_idx)
        remaining.remove(best_idx)
        if not remaining:
            break
        rem2     = np.array(remaining, dtype=np.intp)
        # corr(f_j, f_best) for all j in remaining
        corr_new = np.abs(Xn[:, rem2].T @ Xn[:, best_idx]) / n   # (|rem2|,)
        max_corr_arr[rem2] = np.maximum(max_corr_arr[rem2], corr_new)
        scores[rem2] = corr_y[rem2] / (1.0 + max_corr_arr[rem2])

    # Fill any leftover (shouldn't happen, but safety)
    leftover = [i for i in range(p) if i not in rank_list]
    rank_list.extend(leftover)
    return np.array(rank_list, dtype=np.intp)


def rank_fusedlasso(Xs, y, **_):
    """
    Fused Lasso (Tibshirani et al. 2005) approximation.
    Standard L1-LR ranking + fusion (smoothness) penalty along the feature
    ordering induced by the ANOVA score.  Features are first sorted by ANOVA
    rank; a finite-difference penalty is applied to encourage adjacent (in
    ANOVA rank) features to share similar weights.

    Implementation: proximal gradient on LASSO coefs with fused-TV penalty.
    lambda_fused controls the strength of the smoothness term.
    """
    n, p = Xs.shape
    # Step 1: ANOVA ordering
    F, _ = f_classif(Xs, y)
    F    = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    anova_order = np.argsort(F)[::-1]          # descending relevance

    # Step 2: LASSO coefs on ANOVA-ordered features
    lrcv = LogisticRegressionCV(
        Cs=L1_CV_CS, penalty='l1', solver='saga', cv=3,
        max_iter=200, tol=1e-3, refit=False, n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    lrcv.fit(Xs, y)
    best_C = float(lrcv.C_[0])
    lr = LogisticRegression(
        penalty='l1', C=best_C, solver='saga',
        max_iter=500, tol=1e-4, random_state=RANDOM_STATE,
    )
    lr.fit(Xs, y)
    abs_coef = np.abs(lr.coef_[0])            # (p,)

    # Step 3: 1-D Fused Lasso smoothing via ADMM along ANOVA ordering
    w     = abs_coef[anova_order].copy()    # (p,) reordered weights
    lam_f = max(0.05 * float(w.max()), 1e-9)
    beta  = w.copy()                        # (p,)
    z     = np.diff(beta)                   # (p-1,)
    uu    = np.zeros(p - 1)                 # (p-1,) scaled dual

    def _thomas(d, e, rhs):
        """Tridiagonal solve via Thomas algorithm. d=diag, e=off-diag."""
        n_ = len(d)
        dc = d.copy(); rc = rhs.copy()
        for i in range(1, n_):
            m      = e[i-1] / dc[i-1]
            dc[i] -= m * e[i-1]
            rc[i] -= m * rc[i-1]
        x = np.zeros(n_)
        x[-1] = rc[-1] / dc[-1]
        for i in range(n_-2, -1, -1):
            x[i] = (rc[i] - e[i] * x[i+1]) / dc[i]
        return x

    def _soft_thresh(v, t):
        return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

    # (I + C^T C) is tridiagonal with diag=[1,2,...,2,1], off-diag=-1
    diag    = np.full(p, 2.0); diag[0] = diag[-1] = 1.0
    offdiag = np.full(p - 1, -1.0)

    for _ in range(15):
        rhs       = w.copy()
        rhs[:-1] -= (z - uu)               # C^T (z-u)
        rhs[1:]  += (z - uu)
        beta      = _thomas(diag, offdiag, rhs)
        Cb        = np.diff(beta)           # (p-1,)
        z         = _soft_thresh(Cb + uu, lam_f)
        uu        = uu + Cb - z

    fused_weights = np.zeros(p)
    fused_weights[anova_order] = np.maximum(beta, 0.0)
    return np.argsort(-fused_weights).copy()


def rank_grouplasso(Xs, y, **_):
    """
    Group Lasso (Ma et al. 2007) approximation.
    Features are divided into equal-sized groups of GL_GROUP_SIZE.
    Each group is scored by the L2 norm of the class-mean differences
    (MANOVA-style group score).  Within each group, features are ranked
    by their individual abs-correlation with y.
    """
    n, p   = Xs.shape
    groups = [list(range(i, min(i + GL_GROUP_SIZE, p)))
              for i in range(0, p, GL_GROUP_SIZE)]

    # Class means per group
    mu0 = Xs[y == 0].mean(axis=0)
    mu1 = Xs[y == 1].mean(axis=0)
    diff = mu1 - mu0                           # (p,)

    # Group scores: L2 norm of mean differences within the group
    group_scores = np.array([np.linalg.norm(diff[g]) for g in groups])
    # Sort groups descending
    group_order  = np.argsort(-group_scores)

    # Within each group: rank by abs(mean-diff) = individual feature score
    rank_list = []
    for gi in group_order:
        g = groups[gi]
        local_scores = np.abs(diff[g])
        within_order = np.argsort(-local_scores)
        rank_list.extend([g[j] for j in within_order])

    return np.array(rank_list, dtype=np.intp)


def _interaction_scores(Xs, y):
    """
    Compute diagonal of X^T · Sigma_b · X, where Sigma_b is the
    between-class covariance matrix.  This is the interaction-aware
    relevance signal used by InLasso / InFusedLasso / InElasticNet.

    In the Bai et al. paper, the interaction matrix U is built from
    kernel-graph JSD similarities (Eq. 4).  Here we use the Fisher
    criterion's between-class scatter as a tractable approximation that
    captures the same spirit: features with high between-class variance
    AND high pairwise correlation with discriminative features score highly.

    Returns a (p,) array of interaction scores (non-negative).
    """
    n, p   = Xs.shape
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    mu0    = Xs[y == 0].mean(axis=0)
    mu1    = Xs[y == 1].mean(axis=0)
    mu     = Xs.mean(axis=0)
    # Between-class scatter diagonal
    Sb_diag = (n0 * (mu0 - mu)**2 + n1 * (mu1 - mu)**2) / n
    # Within-class scatter diagonal
    Sw_diag = (Xs[y==0] - mu0).var(axis=0) * n0/n + \
              (Xs[y==1] - mu1).var(axis=0) * n1/n
    # Fisher ratio per feature
    fisher  = Sb_diag / (Sw_diag + 1e-12)
    # Pairwise interaction: score(j) = fisher(j) * mean(|corr(j, top-k)|)
    # Use only top-500 by Fisher for efficiency
    top_k   = min(500, p)
    top_idx = np.argsort(fisher)[-top_k:]
    Xn      = Xs - Xs.mean(axis=0)
    std_X   = np.sqrt((Xn**2).sum(axis=0)) + 1e-12
    Xn     /= std_X
    # Interaction: each feature's mean abs-correlation with top_k fisher features
    cross_corr = np.abs(Xn.T @ Xn[:, top_idx]) / n  # (p, top_k)
    interact   = cross_corr.mean(axis=1) * fisher
    return interact


def rank_inlasso(Xs, y, **_):
    """
    InLasso — Interacted Lasso (Zhang et al. 2017) approximation.
    Rank features by interaction-adjusted LASSO weights:
      score(j) = |lasso_coef(j)| * (1 + alpha * interaction(j))
    where alpha balances the standard LASSO and interaction terms.
    """
    lrcv = LogisticRegressionCV(
        Cs=L1_CV_CS, penalty='l1', solver='saga', cv=3,
        max_iter=200, tol=1e-3, refit=False, n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    lrcv.fit(Xs, y)
    lr = LogisticRegression(
        penalty='l1', C=float(lrcv.C_[0]), solver='saga',
        max_iter=500, tol=1e-4, random_state=RANDOM_STATE,
    )
    lr.fit(Xs, y)
    abs_coef = np.abs(lr.coef_[0])
    interact = _interaction_scores(Xs, y)
    # Normalise interaction to [0,1]
    interact /= (interact.max() + 1e-12)
    alpha     = 0.5
    score     = abs_coef * (1.0 + alpha * interact)
    return np.argsort(-score).copy()


def rank_infusedlasso(Xs, y, **_):
    """
    InFusedLasso — Structural Interacting Fused Lasso (Bai et al. 2019).
    Combines InLasso interaction-adjusted weights with the fused-lasso
    smoothness prior (successive-differences penalty) from rank_fusedlasso.
    Score(j) = fused_weight(j) * (1 + alpha * interaction(j))
    """
    # Step 1: interaction scores
    interact = _interaction_scores(Xs, y)
    interact /= (interact.max() + 1e-12)

    # Step 2: LASSO coefs
    lrcv = LogisticRegressionCV(
        Cs=L1_CV_CS, penalty='l1', solver='saga', cv=3,
        max_iter=200, tol=1e-3, refit=False, n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    lrcv.fit(Xs, y)
    lr = LogisticRegression(
        penalty='l1', C=float(lrcv.C_[0]), solver='saga',
        max_iter=500, tol=1e-4, random_state=RANDOM_STATE,
    )
    lr.fit(Xs, y)
    abs_coef = np.abs(lr.coef_[0])

    # Step 3: ANOVA ordering for fused smoothness
    F, _        = f_classif(Xs, y)
    F           = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    anova_order = np.argsort(F)[::-1]

    w     = abs_coef[anova_order].copy()
    lam_f = max(0.05 * float(w.max()), 1e-9)
    p_    = len(w)
    beta  = w.copy()
    z     = np.diff(beta)
    uu    = np.zeros(p_ - 1)

    def _thomas2(d, e, rhs):
        n_ = len(d)
        dc = d.copy(); rc = rhs.copy()
        for i in range(1, n_):
            m = e[i-1] / dc[i-1]; dc[i] -= m*e[i-1]; rc[i] -= m*rc[i-1]
        x = np.zeros(n_); x[-1] = rc[-1]/dc[-1]
        for i in range(n_-2, -1, -1):
            x[i] = (rc[i] - e[i]*x[i+1]) / dc[i]
        return x

    diag2    = np.full(p_, 2.0); diag2[0] = diag2[-1] = 1.0
    offdiag2 = np.full(p_ - 1, -1.0)

    for _ in range(15):
        rhs       = w.copy()
        rhs[:-1] -= (z - uu)
        rhs[1:]  += (z - uu)
        beta      = _thomas2(diag2, offdiag2, rhs)
        Cb        = np.diff(beta)
        z         = np.sign(Cb + uu) * np.maximum(np.abs(Cb + uu) - lam_f, 0.0)
        uu        = uu + Cb - z

    fused_weights = np.zeros(len(abs_coef))
    fused_weights[anova_order] = np.maximum(beta, 0.0)

    # Step 4: combine
    alpha = 0.5
    score = fused_weights * (1.0 + alpha * interact)
    return np.argsort(-score).copy()


def rank_inelasticnet(Xs, y, **_):
    """
    InElasticNet — Interacted Elastic Net (Cui et al. 2019) approximation.
    L1+L2 penalised Logistic Regression (elastic net) with interaction
    reweighting.  The L2 term stabilises the solution in the high-p regime.
    Score(j) = |en_coef(j)| * (1 + alpha * interaction(j))
    """
    # ElasticNet LR (l1_ratio = 0.5: equal L1 and L2)
    best_score = -np.inf
    best_coef  = None
    for C in L1_CV_CS:
        lr = LogisticRegression(
            penalty='elasticnet', C=C, solver='saga', l1_ratio=0.5,
            max_iter=300, tol=1e-3, random_state=RANDOM_STATE,
        )
        lr.fit(Xs, y)
        # Use training accuracy as proxy for fast C selection
        sc = lr.score(Xs, y)
        if sc > best_score:
            best_score = sc
            best_coef  = lr.coef_[0].copy()

    abs_coef = np.abs(best_coef)
    interact = _interaction_scores(Xs, y)
    interact /= (interact.max() + 1e-12)
    alpha     = 0.5
    score     = abs_coef * (1.0 + alpha * interact)
    return np.argsort(-score).copy()


# ── Ranker registry — all 10 methods ─────────────────────────────────
RANKERS = {
    'LASSO':        rank_lasso,
    'Relief':       rank_relief,
    'ANOVA':        rank_anova,
    'StabSel':      rank_stabsel,
    'ULasso':       rank_ulasso,
    'FusedLasso':   rank_fusedlasso,
    'GroupLasso':   rank_grouplasso,
    'InLasso':      rank_inlasso,
    'InFusedLasso': rank_infusedlasso,
    'InElasticNet': rank_inelasticnet,
}


# ══════════════════════════════════════════════════════════════════════
# 4. STABILITY METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_ki_ji(signatures, p):
    """Kuncheva Index and Jaccard Index over all C(λ,2) pairs."""
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
    """Nogueira Stability Index Ŝ (Nogueira, Brown & Sherrat, JMLR 2018)."""
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
# 5. ONE SHUFFLE
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

        ss_rng = np.random.default_rng(RANDOM_STATE + shuffle_id * 1000 + fold_idx)

        for mname, ranker in RANKERS.items():
            if mname == 'StabSel':
                rank = ranker(Xtr_s, y_tr, rng=ss_rng)
            else:
                rank = ranker(Xtr_s, y_tr)

            for pi, k in enumerate(ks):
                sel = rank[:k]
                clf = LogisticRegressionCV(
                    Cs=CLF_CV_CS, cv=3, penalty='l2', solver='lbfgs',
                    max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE,
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
# 6. FULL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate(X, y, percentages):
    p   = X.shape[1]
    lam = N_SHUFFLES * N_FOLDS
    print(f"  {N_SHUFFLES} shuffles × {N_FOLDS} folds = {lam} signatures per method per %")
    print(f"  {len(RANKERS)} methods: {', '.join(RANKERS.keys())}")

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
# 7. CONSOLE OUTPUT
# ══════════════════════════════════════════════════════════════════════

def print_ki_ji_nogueira_per_percentage(results, percentages):
    p   = 3403
    bar = '═' * 110
    ALL_METHODS = list(RANKERS.keys())

    for mname in ALL_METHODS:
        print(f"\n{bar}")
        print(f"  METHOD: {mname}  —  KI, JI & Nogueira at each % of selected features")
        print(f"  {'%Features':>10}  {'k':>6}  {'KI':>10}  {'JI':>10}  "
              f"{'Nogueira':>10}  {'Accuracy':>10}  {'F1-Score':>10}")
        print(f"{'─'*110}")
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

    col_w = 13
    hdr   = f"  {'%Features':>10}  {'k':>6}" + \
            "".join(f"  {m:>{col_w}}" for m in ALL_METHODS)

    for metric, label in [('ki', 'KUNCHEVA INDEX (KI)'),
                           ('ji', 'JACCARD INDEX (JI)'),
                           ('nogueira', 'NOGUEIRA INDEX (Ŝ)')]:
        print(f"\n{bar}")
        print(f"  {label}  — all methods at each % of selected features")
        print(hdr)
        print(f"{'─'*110}")
        for pi, pct in enumerate(percentages):
            k   = max(1, int(pct / 100.0 * p))
            row = f"  {pct:>10.1f}%  {k:>6d}"
            for m in ALL_METHODS:
                v = results[m][metric][pi]
                row += f"  {v:>{col_w}.4f}" if not np.isnan(v) \
                       else f"  {'N/A':>{col_w}}"
            print(row)
        print(bar)

    print(f"\n{'─'*80}")
    print(f"  Mean KI / JI / Nogueira / Acc / F1  averaged over all {len(percentages)} percentages:")
    print(f"  {'Method':<14}  {'Mean KI':>10}  {'Mean JI':>10}  {'Mean Ŝ':>10}  "
          f"{'Mean Acc':>10}  {'Mean F1':>10}")
    print(f"{'─'*80}")
    for m in ALL_METHODS:
        ki_m = np.nanmean(results[m]['ki'])
        ji_m = np.nanmean(results[m]['ji'])
        ng_v = [v for v in results[m]['nogueira'] if not np.isnan(v)]
        ng_m = np.mean(ng_v) if ng_v else float('nan')
        ac_m = np.mean(results[m]['acc']) * 100
        f1_m = np.mean(results[m]['f1'])  * 100
        ng_s = f"{ng_m:>10.4f}" if not np.isnan(ng_m) else f"{'N/A':>10}"
        print(f"  {m:<14}  {ki_m:>10.4f}  {ji_m:>10.4f}  {ng_s}  "
              f"{ac_m:>9.2f}%  {f1_m:>9.2f}%")
    print(f"{'─'*80}")


def print_tables(results, percentages):
    sep    = '─' * 78
    header = (f"  {'Method':<14}  {'Accuracy':>9}  "
              f"{'Precision':>9}  {'Recall':>9}  {'F1-Score':>9}")
    for pct, tname in [(5.0, 'II  — top  5%'), (10.0, 'III — top 10%')]:
        pi = percentages.index(pct)
        print(f"\n{sep}")
        print(f"  Table {tname} components — Dataset 1 (all 10 methods)")
        print(sep)
        print(header)
        print(sep)
        for m in RANKERS:
            a  = results[m]['acc'][pi] * 100
            pr = results[m]['pre'][pi] * 100
            rc = results[m]['rec'][pi] * 100
            f1 = results[m]['f1'] [pi] * 100
            print(f"  {m:<14}  {a:>8.2f}%  {pr:>8.2f}%  {rc:>8.2f}%  {f1:>8.2f}%")
        print(sep)


# ══════════════════════════════════════════════════════════════════════
# 8. PLOTS — all 10 methods
# ══════════════════════════════════════════════════════════════════════

def _panel(ax, data_dict, pct_list, ylabel, title, methods=None):
    if methods is None:
        methods = list(STYLE.keys())
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for mname in methods:
        if mname in data_dict:
            ax.plot(x, data_dict[mname], label=mname, **STYLE[mname])
    ax.set_xticks(x)
    ax.set_xticklabels(xl, rotation=45, fontsize=7)
    ax.set_xlabel('% of Selected Features', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, loc='left', fontsize=9, pad=4)
    ax.legend(fontsize=6.5, loc='best', framealpha=0.7, ncol=2)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.tick_params(labelsize=7)
    vals = [v for m in methods if m in data_dict
            for v in data_dict[m] if not np.isnan(v)]
    if vals:
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.15, 0.02)
        ax.set_ylim(lo - pad, hi + pad)


def _panel_nogueira_vs_ki(ax, results, pct_list, methods=None):
    if methods is None:
        methods = list(STYLE.keys())
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for mname in methods:
        c  = STYLE[mname]['color']
        ng = [0.0 if np.isnan(v) else v for v in results[mname]['nogueira']]
        ki = results[mname]['ki']
        ax.plot(x, ng, color=c, ls='-',  lw=1.8, ms=4, marker='o',
                label=f'{mname} Ŝ')
        ax.plot(x, ki, color=c, ls='--', lw=1.3, ms=3, marker='s',
                label=f'{mname} KI')
    ax.set_xticks(x)
    ax.set_xticklabels(xl, rotation=45, fontsize=7)
    ax.set_xlabel('% of Selected Features', fontsize=8)
    ax.set_ylabel('Stability Index', fontsize=8)
    ax.set_title('(f) Nogueira Ŝ vs Kuncheva KI', loc='left', fontsize=9, pad=4)
    ax.legend(fontsize=5.5, loc='best', framealpha=0.7, ncol=4)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.tick_params(labelsize=7)


def plot_all_methods(results, percentages):
    """
    Fig. 5 — 2×3 combined panel for ALL 10 methods.
    """
    panels = [
        ('acc',      'Accuracy',        '(a) Accuracy.'),
        ('f1',       'F1 Score',        '(b) F1 Score.'),
        ('ki',       'Kuncheva Index',  '(c) Kuncheva Index.'),
        ('ji',       'Jaccard Index',   '(d) Jaccard Index.'),
        ('nogueira', 'Nogueira Ŝ',      '(e) Nogueira Stability.'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle(
        'All 10 Feature Selection Methods — Dataset 1 (University of Lausanne)\n'
        'Accuracy, F1, Kuncheva KI, Jaccard JI, Nogueira Ŝ\n'
        '54 subjects · 83 ROIs · 3,403 FC features  |  20 shuffles × 5 folds = 100 signatures\n'
        f'[LASSO / Relief / ANOVA / StabSel / ULasso / FusedLasso / GroupLasso / InLasso / InFusedLasso / InElasticNet]',
        fontsize=8, fontweight='bold', y=1.02)

    all_methods = list(STYLE.keys())
    for (key, ylabel, title), ax in zip(panels, axes.flat):
        data_dict = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
                     for m in all_methods}
        _panel(ax, data_dict, percentages, ylabel, title, methods=all_methods)

    _panel_nogueira_vs_ki(axes.flat[5], results, percentages, methods=all_methods)
    plt.tight_layout()
    fig.savefig('fig5_all10_combined.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig5_all10_combined.png")

    # Individual panels
    fnames = {
        'acc':      'fig5_acc.png',
        'f1':       'fig5_f1.png',
        'ki':       'fig5_ki.png',
        'ji':       'fig5_ji.png',
        'nogueira': 'fig5_nogueira.png',
    }
    for key, ylabel, title in panels:
        data_dict = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
                     for m in all_methods}
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        _panel(ax2, data_dict, percentages, ylabel, title, methods=all_methods)
        plt.tight_layout()
        fig2.savefig(fnames[key], dpi=180, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {fnames[key]}")

    fig3, ax3 = plt.subplots(figsize=(12, 7))
    _panel_nogueira_vs_ki(ax3, results, percentages, methods=all_methods)
    ax3.set_title('Nogueira Ŝ vs Kuncheva KI — all 10 methods', fontsize=11)
    plt.tight_layout()
    fig3.savefig('fig5_nogueira_vs_ki.png', dpi=180, bbox_inches='tight')
    plt.close(fig3)
    print("  Saved: fig5_nogueira_vs_ki.png")


def plot_lasso_family(results, percentages):
    """
    Fig. 6 — Lasso family only: LASSO, ULasso, FusedLasso, GroupLasso,
    InLasso, InFusedLasso, InElasticNet.
    """
    lasso_methods = ['LASSO', 'ULasso', 'FusedLasso', 'GroupLasso',
                     'InLasso', 'InFusedLasso', 'InElasticNet']
    panels = [
        ('acc',      'Accuracy',        '(a) Accuracy'),
        ('f1',       'F1 Score',        '(b) F1 Score'),
        ('ki',       'Kuncheva Index',  '(c) Kuncheva KI'),
        ('ji',       'Jaccard Index',   '(d) Jaccard JI'),
        ('nogueira', 'Nogueira Ŝ',      '(e) Nogueira Ŝ'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        'Fig. 6 — Lasso Family Comparison (7 methods)\n'
        'LASSO · ULasso · FusedLasso · GroupLasso · InLasso · InFusedLasso · InElasticNet\n'
        '54 subjects · 83 ROIs · 3,403 FC features',
        fontsize=9, fontweight='bold', y=1.01)

    for (key, ylabel, title), ax in zip(panels, axes.flat):
        data_dict = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
                     for m in lasso_methods}
        _panel(ax, data_dict, percentages, ylabel, title, methods=lasso_methods)

    _panel_nogueira_vs_ki(axes.flat[5], results, percentages, methods=lasso_methods)
    plt.tight_layout()
    fig.savefig('fig6_lasso_family.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig6_lasso_family.png")


def plot_heatmap_summary(results, percentages):
    """
    Fig. 7 — Heat-map summary: rows = methods, cols = metrics at 5% and 10%.
    """
    methods = list(RANKERS.keys())
    pi5  = percentages.index(5.0)
    pi10 = percentages.index(10.0)

    cols = ['Acc@5%', 'F1@5%', 'KI@5%', 'JI@5%', 'Ŝ@5%',
            'Acc@10%', 'F1@10%', 'KI@10%', 'JI@10%', 'Ŝ@10%']
    data = []
    for m in methods:
        row = [
            results[m]['acc'][pi5]  * 100,
            results[m]['f1'] [pi5]  * 100,
            results[m]['ki'] [pi5],
            results[m]['ji'] [pi5],
            results[m]['nogueira'][pi5] if not np.isnan(results[m]['nogueira'][pi5]) else 0.0,
            results[m]['acc'][pi10] * 100,
            results[m]['f1'] [pi10] * 100,
            results[m]['ki'] [pi10],
            results[m]['ji'] [pi10],
            results[m]['nogueira'][pi10] if not np.isnan(results[m]['nogueira'][pi10]) else 0.0,
        ]
        data.append(row)
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(14, 7))
    # Normalise each column to [0,1] for display
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-12)
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_title('Method Performance Heatmap — all 10 methods\n'
                 '(Colour: column-normalised; green = best, red = worst)',
                 fontsize=11, pad=10)

    # Annotate with actual values
    for i in range(len(methods)):
        for j in range(len(cols)):
            v = data[i, j]
            fmt = f"{v:.2f}" if j in (2, 3, 4, 7, 8, 9) else f"{v:.1f}"
            ax.text(j, i, fmt, ha='center', va='center',
                    fontsize=7.5, color='black', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Column-normalised score', shrink=0.8)
    plt.tight_layout()
    fig.savefig('fig7_heatmap.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig7_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    bar = '═' * 80
    print(bar)
    print('  10-Method Feature Selection Evaluation')
    print('  Dataset 1: University of Lausanne  (83 ROIs, n=54, p=3403)')
    print()
    print('  Methods evaluated:')
    print('  ① LASSO        — L1-LR saga (full-data refit)')
    print('  ② Relief       — ReliefF nearest-hit/miss')
    print('  ③ ANOVA        — F-statistic univariate filter')
    print('  ④ StabSel      — Meinshausen & Bühlmann Stability Selection')
    print('  ⑤ ULasso       — Uncorrelated Lasso (Chen et al. 2013)')
    print('  ⑥ FusedLasso   — Fused Lasso (Tibshirani et al. 2005)')
    print('  ⑦ GroupLasso   — Group Lasso (Ma et al. 2007)')
    print('  ⑧ InLasso      — Interacted Lasso (Zhang et al. 2017)')
    print('  ⑨ InFusedLasso — Structural Interacting Fused Lasso (Bai et al. 2019)')
    print('  ⑩ InElasticNet — Interacted Elastic Net (Cui et al. 2019)')
    print()
    print('  Stability metrics: Kuncheva KI · Jaccard JI · Nogueira Ŝ')
    print('  Classifier: LogisticRegressionCV (L2, tuned C) — fair for all methods')
    print(bar)

    X, y, n_rois = load_data()

    print(f'\n[Step 1]  Running {N_SHUFFLES}×{N_FOLDS} cross-validation ...')
    results = evaluate(X, y, PERCENTAGES)

    print('\n[Step 2]  Per-percentage KI, JI, Nogueira for all 10 methods:')
    print_ki_ji_nogueira_per_percentage(results, PERCENTAGES)

    print('\n[Step 3]  Tables II & III (5% and 10% feature subsets):')
    print_tables(results, PERCENTAGES)

    print('\n[Step 4]  Plotting Fig. 5 — all 10 methods combined ...')
    plot_all_methods(results, PERCENTAGES)

    print('\n[Step 5]  Plotting Fig. 6 — Lasso family comparison ...')
    plot_lasso_family(results, PERCENTAGES)

    print('\n[Step 6]  Plotting Fig. 7 — Performance heatmap ...')
    plot_heatmap_summary(results, PERCENTAGES)

    print(f'\n{bar}')
    print('  All done.  Output files:')
    print('    fig5_all10_combined.png  — 2×3 grid, all 10 methods')
    print('    fig5_acc.png, fig5_f1.png, fig5_ki.png, fig5_ji.png, fig5_nogueira.png')
    print('    fig5_nogueira_vs_ki.png  — Ŝ vs KI overlay')
    print('    fig6_lasso_family.png    — 7 Lasso-family methods')
    print('    fig7_heatmap.png         — performance heatmap at 5% and 10%')
    print(bar)


if __name__ == '__main__':
    main()
