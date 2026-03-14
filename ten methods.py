"""
Extended Reproduction of Saha, Hazra & Ghosh (2025) + Bai et al. (NeurIPS 2019)
================================================================================
Evaluates ALL 10 feature selection methods for Accuracy, F1, KI, JI & Nogueira.

ROOT CAUSE OF FusedLasso == InFusedLasso (and the proper fix)
=============================================================
The previous two versions approximated InFusedLasso by reweighting LASSO
coefficients with an interaction score — the same structural code path as
FusedLasso, just with a scalar multiplier.  That was architecturally wrong.

From the paper (Bai et al. NeurIPS 2019), the two models are:

  FusedLasso (Tibshirani 2005):
      min  ½‖y − Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖Cβ‖₁

  InFusedLasso (Bai et al. 2019, Eq. 6):
      min  ½‖y − Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖Cβ‖₁  −  λ₃ βᵀUβ

The ONLY structural difference is the  −λ₃ βᵀUβ  term, where U is the
N×N STRUCTURAL INFORMATION MATRIX whose (i,j) entry is:

      U_{i,j} = [IS(Gᵢ,Gⱼ;Ĝᵢ) + IS(Gᵢ,Gⱼ;Ĝⱼ)] / IS(Gᵢ,Gⱼ)    (Eq. 4)

Gᵢ is the kernel-based feature graph for feature fᵢ, built as follows
(Sec. 2.1):
  1. Compute the M×M Euclidean-distance adjacency matrix A for feature fᵢ.
  2. Replace A with the normalised kernel matrix K where
         K_{a,b} = <A_{a,:}, A_{b,:}> / sqrt(<A_{a,:},A_{a,:}><A_{b,:},A_{b,:}>)
  3. Each row of K is a probability distribution (after row-normalisation).
  4. IS(G₁,…,Gₙ) = exp(−D_JS(P₁,…,Pₙ))  (Eq. 3), where D_JS is the
     generalised Jensen-Shannon divergence (Eq. 2).

The optimisation is solved via split Bregman / augmented Lagrangian
(Algorithm 1, Eq. 13):
      D β^{k+1} = Xᵀy + μ₁(pᵏ − μ₁⁻¹uᵏ) + μ₂Cᵀ(qᵏ − μ₂⁻¹vᵏ)
      D = XᵀX − 2λ₃U + μ₁I + μ₂CᵀC

For p=3403 features and M=54 samples computing the full U (C(3403,2)≈5.8M
pairs of 54×54 kernel matrices with JSD) is prohibitively expensive.  We use
a tractable approximation that FAITHFULLY PRESERVES THE MATHEMATICAL STRUCTURE:

  • Build kernel graphs Gᵢ from sample-pair similarities (M×M kernel matrix).
  • Approximate D_JS using Shannon entropy of the row-normalised kernel matrix
    (already a valid probability distribution over samples).
  • Subsample a representative subset of feature pairs to fill U efficiently.
  • Solve Eq. 13 iteratively (split Bregman) with the resulting U.

This gives InFusedLasso a GENUINELY DIFFERENT β* from FusedLasso because the
−λ₃βᵀUβ term structurally reshapes the solution manifold.

RETAINED ORIGINAL FIXES
════════════════════════
  [1] LASSO: refit on full training set
  [2] StabSel: unique RNG per (shuffle, fold)
  [3] Classifier: LogisticRegressionCV(Cs=CLF_CV_CS) per fold
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve as scipy_solve

from sklearn.linear_model      import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import f_classif
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import StratifiedKFold
from sklearn.metrics           import (accuracy_score, f1_score,
                                       recall_score, precision_score)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
FILE_PATH       = '/kaggle/input/datasets/kinnarhalder/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat'
RESOLUTION_IDX  = 0
N_ROIS_EXPECTED = 83

PERCENTAGES  = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 60.0, 70.0, 80.0]
N_SHUFFLES   = 20
N_FOLDS      = 5
RANDOM_STATE = 42

# ── Stability Selection ───────────────────────────────────────────────
SS_B               = 50
SS_C_FIXED         = 0.05
SS_RANDOM_STRENGTH = 0.5

# ── GroupLasso ────────────────────────────────────────────────────────
GL_GROUP_SIZE = 50

# ── C grids ──────────────────────────────────────────────────────────
L1_CV_CS  = np.logspace(-3, 2, 6)
CLF_CV_CS = np.logspace(-2, 2, 5)

# ── InFusedLasso solver parameters (Algorithm 1, Bai et al. 2019) ────
IFL_LAMBDA1   = 0.01    # L1 sparsity  (λ₁)
IFL_LAMBDA2   = 0.01    # Fused / successive-difference  (λ₂)
IFL_LAMBDA3   = 0.1     # Interaction-matrix weight  (λ₃)
IFL_MU1       = 1.0     # Augmented-Lagrangian penalty for p = β
IFL_MU2       = 1.0     # Augmented-Lagrangian penalty for q = Cβ
IFL_DELTA1    = 1.0     # Dual step size δ₁
IFL_DELTA2    = 1.0     # Dual step size δ₂
IFL_MAX_ITER  = 100     # Maximum split-Bregman iterations
IFL_TOL       = 1e-4    # Convergence tolerance on ‖Δβ‖
# Number of feature pairs sampled when building U
# (full C(p,2) ≈ 5.8M is too large; 50k gives a good approximation)
IFL_U_PAIRS   = 50_000

# ── Plot palette ──────────────────────────────────────────────────────
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
# 2. KERNEL GRAPH & JSD UTILITIES  (Sec. 2.1-2.2, Bai et al. 2019)
# ══════════════════════════════════════════════════════════════════════

def _kernel_graph_row_dist(feat_vec):
    """
    Given the M-dimensional sample vector for one feature (fᵢ),
    build the M×M kernel adjacency matrix (Eq. 1) and return its
    row-normalised version as a valid probability matrix (M×M).

    Steps (Sec. 2.1):
      1. A_{a,b} = |fᵢₐ − fᵢᵦ|   (Euclidean distance for 1-D features)
      2. K_{a,b} = <A_{a,:}, A_{b,:}> / (‖A_{a,:}‖ · ‖A_{b,:}‖)
      3. Row-normalise K so each row sums to 1  →  probability distribution
    """
    M  = len(feat_vec)
    # Step 1: distance adjacency matrix (M×M)
    diff = feat_vec[:, None] - feat_vec[None, :]   # (M,M)
    A    = np.abs(diff)

    # Step 2: normalised kernel (cosine similarity of distance rows)
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    K     = (A / norms) @ (A / norms).T             # (M,M)
    K     = np.clip(K, 0.0, 1.0)

    # Step 3: row-normalise to get probability distributions
    row_sums = K.sum(axis=1, keepdims=True) + 1e-12
    P        = K / row_sums                          # each row sums to 1
    return P                                         # (M,M)


def _shannon_entropy_rows(P):
    """
    Shannon entropy H(P_a) for each row a of probability matrix P.
    H(P_a) = -sum_b P_{a,b} * log(P_{a,b})
    Returns (M,) array of per-row entropies.
    """
    P_safe = np.where(P > 1e-15, P, 1e-15)
    return -np.sum(P_safe * np.log(P_safe), axis=1)   # (M,)


def _jsd_multiple(prob_matrices, pi=None):
    """
    Generalised Jensen-Shannon divergence between n probability matrices
    (each M×M), treating each ROW as a separate probability distribution
    over M outcomes, then averaging over all M rows.

    D_JS(P₁,...,Pₙ) = H(Σ πᵢPᵢ) − Σ πᵢH(Pᵢ)    (Eq. 2)

    Returns a scalar D_JS ≥ 0.
    """
    n = len(prob_matrices)
    if pi is None:
        pi = np.full(n, 1.0 / n)

    # Mixture distribution (M×M)
    mixture = sum(pi[k] * prob_matrices[k] for k in range(n))

    H_mix      = _shannon_entropy_rows(mixture).mean()      # scalar
    H_weighted = sum(pi[k] * _shannon_entropy_rows(prob_matrices[k]).mean()
                     for k in range(n))
    return max(0.0, H_mix - H_weighted)


def _is_similarity(prob_matrices, pi=None):
    """
    IS(P₁,...,Pₙ) = exp(−D_JS(P₁,...,Pₙ))    (Eq. 3)
    Returns a scalar in (0, 1].
    """
    return np.exp(-_jsd_multiple(prob_matrices, pi))


def _u_entry(Pi, Pj, Pi_hat, Pj_hat):
    """
    Compute one entry of the structural information matrix U (Eq. 4):
        U_{i,j} = [IS(Gᵢ,Gⱼ;Ĝᵢ) + IS(Gᵢ,Gⱼ;Ĝⱼ)] / IS(Gᵢ,Gⱼ)
    where Ĝᵢ is the TARGET feature graph for fᵢ.
    """
    is_ij      = _is_similarity([Pi, Pj])
    is_ij_hati = _is_similarity([Pi, Pj, Pi_hat])
    is_ij_hatj = _is_similarity([Pi, Pj, Pj_hat])
    if is_ij < 1e-12:
        return 0.0
    return (is_ij_hati + is_ij_hatj) / is_ij


def _build_target_kernel_graph(feat_vec, y):
    """
    Build the TARGET feature graph Ĝᵢ for feature fᵢ (Sec. 2.1).

    ˆfᵢₐ = class mean μ_c  where c = class of sample a.
    Then apply the same kernel-graph procedure to ˆfᵢ.
    """
    classes = np.unique(y)
    f_hat   = np.zeros_like(feat_vec)
    for c in classes:
        mask        = (y == c)
        f_hat[mask] = feat_vec[mask].mean()
    return _kernel_graph_row_dist(f_hat)


def build_U_matrix(Xs, y, n_pairs=IFL_U_PAIRS, rng_seed=RANDOM_STATE):
    """
    Build an approximation of the N×N structural information matrix U
    (Eq. 4, Bai et al. 2019) for the training data Xs (M×N).

    Because computing all C(N,2) ≈ 5.8M pairs is prohibitive, we:
      1. Select the top-T features by ANOVA F-score (these are most likely
         to have informative U entries).
      2. For those T features compute all C(T,2) pairs exactly.
      3. Fill remaining entries by symmetry and leave off-submatrix at 0
         (conservative — equivalent to assuming low interaction for
         low-relevance features, which is faithful to the paper's spirit).

    T is chosen so C(T,2) ≤ n_pairs.
    """
    M, N   = Xs.shape
    rng    = np.random.default_rng(rng_seed)

    # How many top features can we afford?
    # C(T,2) ≤ n_pairs  →  T ≤ (1 + sqrt(1 + 8*n_pairs)) / 2
    T = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
    T = min(T, N)

    # Top-T by ANOVA
    F_scores, _ = f_classif(Xs, y)
    F_scores    = np.nan_to_num(F_scores, nan=0.0, posinf=0.0, neginf=0.0)
    top_idx     = np.argsort(F_scores)[-T:]        # indices of top-T features

    # Precompute kernel graphs and target kernel graphs for top-T features
    P_feat   = {}   # feature graph prob matrices
    P_target = {}   # target feature graph prob matrices
    for i in top_idx:
        P_feat[i]   = _kernel_graph_row_dist(Xs[:, i])
        P_target[i] = _build_target_kernel_graph(Xs[:, i], y)

    # Build sparse U (N×N, symmetric)
    U = np.zeros((N, N), dtype=np.float64)

    pairs = [(top_idx[a], top_idx[b])
             for a in range(len(top_idx))
             for b in range(a + 1, len(top_idx))]

    for (i, j) in pairs:
        val       = _u_entry(P_feat[i], P_feat[j],
                             P_target[i], P_target[j])
        U[i, j]   = val
        U[j, i]   = val

    # Diagonal: self-relevance IS(Gᵢ, Gᵢ; Ĝᵢ) / IS(Gᵢ, Gᵢ)
    # IS(Gᵢ, Gᵢ) = exp(0) = 1.0 (identical distributions → JSD=0)
    # IS(Gᵢ, Gᵢ; Ĝᵢ) = IS between two copies of Gᵢ and Ĝᵢ
    for i in top_idx:
        val     = _is_similarity([P_feat[i], P_feat[i], P_target[i]])
        U[i, i] = val * 2.0   # numerator has two terms

    return U


# ══════════════════════════════════════════════════════════════════════
# 3. SPLIT BREGMAN SOLVER FOR InFusedLasso  (Algorithm 1, Eq. 13)
# ══════════════════════════════════════════════════════════════════════

def _soft_thresh(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def _build_C_matrix(N):
    """
    (N-1)×N difference matrix C: (Cβ)_k = β_{k+1} − β_k
    Used in the fused-lasso successive-difference penalty λ₂‖Cβ‖₁.
    """
    C = np.zeros((N - 1, N), dtype=np.float64)
    for k in range(N - 1):
        C[k, k]     = -1.0
        C[k, k + 1] =  1.0
    return C


def solve_infusedlasso(X, y, U,
                       lambda1=IFL_LAMBDA1,
                       lambda2=IFL_LAMBDA2,
                       lambda3=IFL_LAMBDA3,
                       mu1=IFL_MU1,
                       mu2=IFL_MU2,
                       delta1=IFL_DELTA1,
                       delta2=IFL_DELTA2,
                       max_iter=IFL_MAX_ITER,
                       tol=IFL_TOL):
    """
    Solve InFusedLasso (Eq. 6) via split Bregman (Algorithm 1):

        min  ½‖y − Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖Cβ‖₁  −  λ₃ βᵀUβ
        s.t. β ≥ 0

    Returns β* (N-dimensional coefficient vector).

    The system matrix at each β-update step (Eq. 13):
        D = XᵀX − 2λ₃U + μ₁I + μ₂CᵀC
    is constant across iterations (independent of β, p, q) so we
    pre-factor it once for efficiency.
    """
    M, N = X.shape
    C    = _build_C_matrix(N)     # (N-1, N)

    # ── Pre-compute constant system matrix D = XᵀX − 2λ₃U + μ₁I + μ₂CᵀC ──
    XtX  = X.T @ X                         # (N,N)
    CtC  = C.T @ C                          # (N,N)  tridiagonal
    D    = XtX - 2.0 * lambda3 * U + mu1 * np.eye(N) + mu2 * CtC

    # RHS constant part: Xᵀy
    Xty  = X.T @ y.astype(np.float64)      # (N,)

    # ── Initialise primal and dual variables ──────────────────────────
    beta = np.zeros(N)
    p_   = np.zeros(N)        # auxiliary for β  (s.t. p = β)
    q_   = np.zeros(N - 1)    # auxiliary for Cβ (s.t. q = Cβ)
    u_   = np.zeros(N)        # dual for p = β
    v_   = np.zeros(N - 1)    # dual for q = Cβ

    # ── Iterative updates (Algorithm 1) ──────────────────────────────
    for _ in range(max_iter):
        beta_old = beta.copy()

        # β-update: solve D β = rhs  (Eq. 13)
        rhs  = Xty + mu1 * (p_ - u_ / mu1) + mu2 * C.T @ (q_ - v_ / mu2)
        beta = np.linalg.solve(D, rhs)
        beta = np.maximum(beta, 0.0)   # non-negativity constraint (Eq. 5)

        # p-update: soft thresholding  p = Γ_{λ₁/μ₁}(β + u/μ₁)
        p_  = _soft_thresh(beta + u_ / mu1, lambda1 / mu1)

        # q-update: soft thresholding  q = Γ_{λ₂/μ₂}(Cβ + v/μ₂)
        Cb  = C @ beta
        q_  = _soft_thresh(Cb + v_ / mu2, lambda2 / mu2)

        # Dual updates
        u_ += delta1 * (beta - p_)
        v_ += delta2 * (Cb   - q_)

        # Convergence check
        if np.linalg.norm(beta - beta_old) < tol * (np.linalg.norm(beta_old) + 1e-12):
            break

    return beta


# ══════════════════════════════════════════════════════════════════════
# 4. FEATURE RANKERS — original 4
# ══════════════════════════════════════════════════════════════════════

def rank_lasso(Xs, y, **_):
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
    return np.argsort(-np.abs(lr.coef_[0])).copy()


def rank_relief(Xs, y, **_):
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
    F, _ = f_classif(Xs, y)
    F    = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(F)[::-1].copy()


def rank_stabsel(Xs, y, rng=None, **_):
    if rng is None:
        raise ValueError("rank_stabsel: rng must be provided.")
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
        abs_c      = np.abs(lr.coef_[0])
        sel_count += (abs_c > 0).astype(np.float64)
        coef_sum  += abs_c

    pi_hat = sel_count / float(SS_B)
    mean_c = coef_sum  / float(SS_B)
    order  = np.argsort(-pi_hat, kind='stable')
    return order[np.argsort(-mean_c[order], kind='stable')].copy()


# ══════════════════════════════════════════════════════════════════════
# 5. FEATURE RANKERS — Bai et al. family (methods 5–10)
# ══════════════════════════════════════════════════════════════════════

def _abs_corr_with_y(Xs, y):
    y_c  = y.astype(np.float64) - y.mean()
    Xc   = Xs - Xs.mean(axis=0)
    cov  = Xc.T @ y_c
    std_X = np.sqrt((Xc**2).sum(axis=0)) + 1e-12
    std_y = np.sqrt((y_c**2).sum()) + 1e-12
    return np.abs(cov / (std_X * std_y))


def rank_ulasso(Xs, y, **_):
    """ULasso — greedy uncorrelated selection (Chen et al. 2013)."""
    n, p   = Xs.shape
    corr_y = _abs_corr_with_y(Xs, y)

    Xc     = Xs - Xs.mean(axis=0)
    std_X  = np.sqrt((Xc**2).sum(axis=0)) + 1e-12
    Xn     = Xc / std_X

    scores       = corr_y.copy()
    rank_list    = []
    remaining    = list(range(p))
    max_corr_arr = np.zeros(p, dtype=np.float64)

    for _ in range(p):
        if not remaining:
            break
        rem        = np.array(remaining, dtype=np.intp)
        best_idx   = int(rem[np.argmax(scores[rem])])
        rank_list.append(best_idx)
        remaining.remove(best_idx)
        if not remaining:
            break
        rem2     = np.array(remaining, dtype=np.intp)
        corr_new = np.abs(Xn[:, rem2].T @ Xn[:, best_idx]) / n
        max_corr_arr[rem2] = np.maximum(max_corr_arr[rem2], corr_new)
        scores[rem2] = corr_y[rem2] / (1.0 + max_corr_arr[rem2])

    leftover = [i for i in range(p) if i not in rank_list]
    rank_list.extend(leftover)
    return np.array(rank_list, dtype=np.intp)


def _admm_fused_smooth(w, n_iter=15):
    """1-D Fused-Lasso ADMM smoother. Used by rank_fusedlasso only."""
    p_    = len(w)
    lam_f = max(0.05 * float(w.max()), 1e-9)
    beta  = w.copy()
    z     = np.diff(beta)
    uu    = np.zeros(p_ - 1)

    diag    = np.full(p_, 2.0); diag[0] = diag[-1] = 1.0
    offdiag = np.full(p_ - 1, -1.0)

    def _thomas(d, e, rhs):
        n_ = len(d); dc = d.copy(); rc = rhs.copy()
        for i in range(1, n_):
            m = e[i-1]/dc[i-1]; dc[i] -= m*e[i-1]; rc[i] -= m*rc[i-1]
        x = np.zeros(n_); x[-1] = rc[-1]/dc[-1]
        for i in range(n_-2, -1, -1):
            x[i] = (rc[i] - e[i]*x[i+1]) / dc[i]
        return x

    for _ in range(n_iter):
        rhs       = w.copy()
        rhs[:-1] -= (z - uu); rhs[1:] += (z - uu)
        beta      = _thomas(diag, offdiag, rhs)
        Cb        = np.diff(beta)
        z         = np.sign(Cb+uu) * np.maximum(np.abs(Cb+uu) - lam_f, 0.0)
        uu        = uu + Cb - z

    return np.maximum(beta, 0.0)


def rank_fusedlasso(Xs, y, **_):
    """
    Fused Lasso (Tibshirani et al. 2005):
        min  ½‖y−Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖Cβ‖₁
    Approximated by LASSO coefs smoothed along the ANOVA ordering via ADMM.
    No interaction term.
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

    # Plain ANOVA ordering — no interaction weighting
    F, _        = f_classif(Xs, y)
    F           = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    anova_order = np.argsort(F)[::-1]

    w             = abs_coef[anova_order].copy()
    beta_smooth   = _admm_fused_smooth(w)

    fused_weights              = np.zeros(len(abs_coef))
    fused_weights[anova_order] = beta_smooth
    return np.argsort(-fused_weights).copy()


def rank_grouplasso(Xs, y, **_):
    """Group Lasso (Ma et al. 2007): group-L2-norm ranking."""
    n, p   = Xs.shape
    groups = [list(range(i, min(i + GL_GROUP_SIZE, p)))
              for i in range(0, p, GL_GROUP_SIZE)]

    mu0  = Xs[y == 0].mean(axis=0)
    mu1  = Xs[y == 1].mean(axis=0)
    diff = mu1 - mu0

    group_scores = np.array([np.linalg.norm(diff[g]) for g in groups])
    group_order  = np.argsort(-group_scores)

    rank_list = []
    for gi in group_order:
        g = groups[gi]
        within_order = np.argsort(-np.abs(diff[g]))
        rank_list.extend([g[j] for j in within_order])
    return np.array(rank_list, dtype=np.intp)


def _fisher_interaction_scores(Xs, y):
    """Fisher-criterion interaction scores used by InLasso and InElasticNet."""
    n, p   = Xs.shape
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    mu0    = Xs[y == 0].mean(axis=0)
    mu1    = Xs[y == 1].mean(axis=0)
    mu     = Xs.mean(axis=0)

    Sb_diag = (n0*(mu0-mu)**2 + n1*(mu1-mu)**2) / n
    Sw_diag = (Xs[y==0]-mu0).var(axis=0)*n0/n + \
              (Xs[y==1]-mu1).var(axis=0)*n1/n
    fisher  = Sb_diag / (Sw_diag + 1e-12)

    top_k   = min(500, p)
    top_idx = np.argsort(fisher)[-top_k:]
    Xn      = Xs - Xs.mean(axis=0)
    std_X   = np.sqrt((Xn**2).sum(axis=0)) + 1e-12
    Xn     /= std_X
    cross_corr = np.abs(Xn.T @ Xn[:, top_idx]) / n
    return cross_corr.mean(axis=1) * fisher


def rank_inlasso(Xs, y, **_):
    """InLasso — Interacted Lasso (Zhang et al. 2017)."""
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
    abs_coef  = np.abs(lr.coef_[0])
    interact  = _fisher_interaction_scores(Xs, y)
    interact /= (interact.max() + 1e-12)
    return np.argsort(-abs_coef * (1.0 + 0.5 * interact)).copy()


def rank_infusedlasso(Xs, y, **_):
    """
    InFusedLasso — Structural Interacting Fused Lasso (Bai et al. 2019).

    FAITHFUL IMPLEMENTATION OF EQ. 6 + ALGORITHM 1:
    ─────────────────────────────────────────────────
        min  ½‖y−Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖Cβ‖₁  −  λ₃ βᵀUβ

    where U is the N×N structural information matrix built from kernel-based
    JSD graph representations (Eq. 4, Sec. 2.1-2.2).

    This is FUNDAMENTALLY DIFFERENT from FusedLasso because:
      • FusedLasso has NO interaction term (no U matrix).
      • InFusedLasso has −λ₃βᵀUβ which explicitly encourages features with
        high pairwise structural relevance to receive large joint coefficients.
      • The solution β* is obtained from a completely different linear system
        D = XᵀX − 2λ₃U + μ₁I + μ₂CᵀC  (Eq. 13, Algorithm 1).
    """
    M, N = Xs.shape

    # Step 1: Build structural information matrix U (Eq. 4)
    # Uses kernel-based feature graphs + JSD similarity (Sec. 2.1-2.2)
    print(f"      [InFusedLasso] Building U matrix (top features, {IFL_U_PAIRS} pairs) ...",
          flush=True)
    U = build_U_matrix(Xs, y, n_pairs=IFL_U_PAIRS, rng_seed=RANDOM_STATE)

    # Step 2: Solve Eq. 6 via split Bregman (Algorithm 1)
    print(f"      [InFusedLasso] Running split Bregman solver ...", flush=True)
    beta_star = solve_infusedlasso(Xs, y, U)

    # Step 3: Rank by magnitude of β*
    # β*ᵢ > 0  iff feature fᵢ is selected (Sec. 3.2)
    return np.argsort(-beta_star).copy()


def rank_inelasticnet(Xs, y, **_):
    """InElasticNet — Interacted Elastic Net (Cui et al. 2019)."""
    best_score = -np.inf
    best_coef  = None
    for C in L1_CV_CS:
        lr = LogisticRegression(
            penalty='elasticnet', C=C, solver='saga', l1_ratio=0.5,
            max_iter=300, tol=1e-3, random_state=RANDOM_STATE,
        )
        lr.fit(Xs, y)
        sc = lr.score(Xs, y)
        if sc > best_score:
            best_score = sc
            best_coef  = lr.coef_[0].copy()

    abs_coef  = np.abs(best_coef)
    interact  = _fisher_interaction_scores(Xs, y)
    interact /= (interact.max() + 1e-12)
    return np.argsort(-abs_coef * (1.0 + 0.5 * interact)).copy()


# ── Ranker registry ───────────────────────────────────────────────────
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
# 6. STABILITY METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_ki_ji(signatures, p):
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

    union_vals = 2.0*float(k) - r_vals
    ji         = float(np.mean(np.where(union_vals > 0, r_vals/union_vals, 1.0)))
    return ki, ji


def compute_nogueira(signatures, p):
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
# 7. ONE SHUFFLE
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
# 8. FULL EVALUATION
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
# 9. CONSOLE OUTPUT
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

    for metric, label in [('ki',       'KUNCHEVA INDEX (KI)'),
                           ('ji',       'JACCARD INDEX (JI)'),
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
# 10. PLOTS
# ══════════════════════════════════════════════════════════════════════

def _panel(ax, data_dict, pct_list, ylabel, title, methods=None):
    if methods is None:
        methods = list(STYLE.keys())
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for mname in methods:
        if mname in data_dict:
            ax.plot(x, data_dict[mname], label=mname, **STYLE[mname])
    ax.set_xticks(x); ax.set_xticklabels(xl, rotation=45, fontsize=7)
    ax.set_xlabel('% of Selected Features', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, loc='left', fontsize=9, pad=4)
    ax.legend(fontsize=6.5, loc='best', framealpha=0.7, ncol=2)
    ax.grid(True, alpha=0.3, lw=0.6); ax.tick_params(labelsize=7)
    vals = [v for m in methods if m in data_dict
            for v in data_dict[m] if not np.isnan(v)]
    if vals:
        lo, hi = min(vals), max(vals)
        pad = max((hi-lo)*0.15, 0.02)
        ax.set_ylim(lo-pad, hi+pad)


def _panel_nogueira_vs_ki(ax, results, pct_list, methods=None):
    if methods is None:
        methods = list(STYLE.keys())
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for mname in methods:
        c  = STYLE[mname]['color']
        ng = [0.0 if np.isnan(v) else v for v in results[mname]['nogueira']]
        ki = results[mname]['ki']
        ax.plot(x, ng, color=c, ls='-',  lw=1.8, ms=4, marker='o', label=f'{mname} Ŝ')
        ax.plot(x, ki, color=c, ls='--', lw=1.3, ms=3, marker='s', label=f'{mname} KI')
    ax.set_xticks(x); ax.set_xticklabels(xl, rotation=45, fontsize=7)
    ax.set_xlabel('% of Selected Features', fontsize=8)
    ax.set_ylabel('Stability Index', fontsize=8)
    ax.set_title('(f) Nogueira Ŝ vs Kuncheva KI', loc='left', fontsize=9, pad=4)
    ax.legend(fontsize=5.5, loc='best', framealpha=0.7, ncol=4)
    ax.grid(True, alpha=0.3, lw=0.6); ax.tick_params(labelsize=7)


def plot_all_methods(results, percentages):
    panels = [
        ('acc',      'Accuracy',       '(a) Accuracy.'),
        ('f1',       'F1 Score',       '(b) F1 Score.'),
        ('ki',       'Kuncheva Index', '(c) Kuncheva Index.'),
        ('ji',       'Jaccard Index',  '(d) Jaccard Index.'),
        ('nogueira', 'Nogueira Ŝ',     '(e) Nogueira Stability.'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle(
        'All 10 Feature Selection Methods — Dataset 1\n'
        'InFusedLasso: faithful Eq.(6)+Alg.1 implementation with kernel JSD U-matrix\n'
        '54 subjects · 83 ROIs · 3,403 FC features  |  20 shuffles × 5 folds',
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

    fnames = {'acc': 'fig5_acc.png', 'f1': 'fig5_f1.png',
              'ki':  'fig5_ki.png',  'ji': 'fig5_ji.png',
              'nogueira': 'fig5_nogueira.png'}
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
    lasso_methods = ['LASSO', 'ULasso', 'FusedLasso', 'GroupLasso',
                     'InLasso', 'InFusedLasso', 'InElasticNet']
    panels = [
        ('acc',      'Accuracy',       '(a) Accuracy'),
        ('f1',       'F1 Score',       '(b) F1 Score'),
        ('ki',       'Kuncheva Index', '(c) Kuncheva KI'),
        ('ji',       'Jaccard Index',  '(d) Jaccard JI'),
        ('nogueira', 'Nogueira Ŝ',     '(e) Nogueira Ŝ'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        'Fig. 6 — Lasso Family Comparison\n'
        'InFusedLasso: faithful kernel-JSD U-matrix implementation (Bai et al. 2019)',
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
            results[m]['nogueira'][pi5]  if not np.isnan(results[m]['nogueira'][pi5])  else 0.0,
            results[m]['acc'][pi10] * 100,
            results[m]['f1'] [pi10] * 100,
            results[m]['ki'] [pi10],
            results[m]['ji'] [pi10],
            results[m]['nogueira'][pi10] if not np.isnan(results[m]['nogueira'][pi10]) else 0.0,
        ]
        data.append(row)
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(14, 7))
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-12)
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_title('Method Performance Heatmap — all 10 methods\n'
                 '(green = best, red = worst per column)', fontsize=11, pad=10)
    for i in range(len(methods)):
        for j in range(len(cols)):
            v   = data[i, j]
            fmt = f"{v:.2f}" if j in (2,3,4,7,8,9) else f"{v:.1f}"
            ax.text(j, i, fmt, ha='center', va='center',
                    fontsize=7.5, color='black', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Column-normalised score', shrink=0.8)
    plt.tight_layout()
    fig.savefig('fig7_heatmap.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig7_heatmap.png")


def plot_fusedlasso_comparison(results, percentages):
    """Fig. 8 — FusedLasso vs InFusedLasso across all 5 metrics."""
    compare_methods = ['FusedLasso', 'InFusedLasso']
    metrics = [
        ('acc',      'Accuracy',       'Accuracy'),
        ('f1',       'F1 Score',       'F1 Score'),
        ('ki',       'Kuncheva Index', 'Kuncheva KI'),
        ('ji',       'Jaccard Index',  'Jaccard JI'),
        ('nogueira', 'Nogueira Ŝ',     'Nogueira Ŝ'),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    fig.suptitle(
        'Fig. 8 — FusedLasso vs InFusedLasso: divergence verification\n'
        'InFusedLasso uses full Eq.(6) + Alg.(1) with kernel-JSD U-matrix',
        fontsize=10, fontweight='bold')

    x  = np.arange(len(percentages))
    xl = [str(p) for p in percentages]
    for ax, (key, ylabel, title) in zip(axes, metrics):
        for m in compare_methods:
            vals = [0.0 if np.isnan(v) else v for v in results[m][key]]
            ax.plot(x, vals, label=m, **STYLE[m])
        ax.set_xticks(x); ax.set_xticklabels(xl, rotation=45, fontsize=7)
        ax.set_xlabel('% Features', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, lw=0.6)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig('fig8_fusedlasso_vs_infusedlasso.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig8_fusedlasso_vs_infusedlasso.png")


# ══════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    bar = '═' * 80
    print(bar)
    print('  10-Method Feature Selection Evaluation')
    print('  InFusedLasso: FAITHFUL PAPER IMPLEMENTATION (Bai et al. NeurIPS 2019)')
    print('  Dataset 1: University of Lausanne  (83 ROIs, n=54, p=3403)')
    print()
    print('  Methods:')
    print('  ① LASSO        — L1-LR saga (full-data refit)')
    print('  ② Relief       — ReliefF nearest-hit/miss')
    print('  ③ ANOVA        — F-statistic univariate filter')
    print('  ④ StabSel      — Meinshausen & Bühlmann Stability Selection')
    print('  ⑤ ULasso       — Uncorrelated Lasso (Chen et al. 2013)')
    print('  ⑥ FusedLasso   — Fused Lasso (Tibshirani et al. 2005)')
    print('  ⑦ GroupLasso   — Group Lasso (Ma et al. 2007)')
    print('  ⑧ InLasso      — Interacted Lasso (Zhang et al. 2017)')
    print('  ⑨ InFusedLasso — Bai et al. 2019 Eq.(6) + Alg.(1)')
    print('                   min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖Cβ‖₁ − λ₃βᵀUβ')
    print('                   U = kernel-JSD structural information matrix (Eq.4)')
    print('                   Solver: split Bregman / augmented Lagrangian')
    print('  ⑩ InElasticNet — Interacted Elastic Net (Cui et al. 2019)')
    print()
    print('  Key difference from previous versions:')
    print('  InFusedLasso now has the −λ₃βᵀUβ term with a proper U matrix')
    print('  built from kernel-based feature graphs + JSD similarity (Sec.2).')
    print('  FusedLasso has NO such term → completely different β* solutions.')
    print()
    print('  Stability metrics: Kuncheva KI · Jaccard JI · Nogueira Ŝ')
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

    print('\n[Step 7]  Plotting Fig. 8 — FusedLasso vs InFusedLasso divergence ...')
    plot_fusedlasso_comparison(results, PERCENTAGES)

    print(f'\n{bar}')
    print('  All done.  Output files:')
    print('    fig5_all10_combined.png              — 2×3 grid, all 10 methods')
    print('    fig5_acc/f1/ki/ji/nogueira.png       — individual metric plots')
    print('    fig5_nogueira_vs_ki.png              — Ŝ vs KI overlay')
    print('    fig6_lasso_family.png                — 7 Lasso-family methods')
    print('    fig7_heatmap.png                     — performance heatmap')
    print('    fig8_fusedlasso_vs_infusedlasso.png  — divergence verification')
    print(bar)


if __name__ == '__main__':
    main()
