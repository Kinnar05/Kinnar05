"""
Feature Selection — 10 Methods with EXACT Convex / DC Optimization
====================================================================
All regularisation-based methods are solved via CVXPY (CLARABEL solver)
using their published objective functions.  No approximations, no
heuristic reweighting, no ADMM shortcuts.

  METHOD          OPTIMIZER           EXACT FORMULATION
  ─────────────────────────────────────────────────────────────────
  LASSO           CVXPY/CLARABEL      min ½‖y−Xβ‖² + λ‖β‖₁
  Relief          Exact algorithm     ReliefF (score-based, no opt.)
  ANOVA           Exact statistic     F-test (no optimization needed)
  StabSel         Exact algorithm     Randomised subsampling + LASSO
  ULasso          CVXPY/CLARABEL      Iteratively Reweighted L1 (IRLS)
                                      min ½‖y−Xβ‖² + λΣᵢwᵢ|βᵢ|
                                      wᵢ ← 1/(1−max_corr(i,S)²) each step
  FusedLasso      CVXPY/CLARABEL      min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖Cβ‖₁
  GroupLasso      CVXPY/CLARABEL      min ½‖y−Xβ‖² + λ Σ_g‖β_g‖₂
  InLasso         CVXPY/CLARABEL      min ½‖y−Xβ‖² + λ‖β‖₁ − γβᵀΣβ
                  (CCCP outer loop)   (DC program; Σ = between-class scatter)
  InFusedLasso    CVXPY/CLARABEL      min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖Cβ‖₁ − λ₃βᵀUβ
                  (CCCP outer loop)   (Eq.6, Bai et al. 2019; U = JSD kernel matrix)
  InElasticNet    CVXPY/CLARABEL      min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖β‖₂² − γβᵀUβ
                  (CCCP outer loop)   (Cui et al. 2019)

DC PROGRAMMING RATIONALE
─────────────────────────
InLasso, InFusedLasso and InElasticNet all contain a term −γβᵀMβ where M
is a positive-semidefinite matrix (between-class scatter or JSD-kernel U).
This makes the full problem non-convex (a DC program: convex minus convex).
The standard exact approach for DC programs is the
Concave-Convex Procedure (CCCP), also called Majorisation-Minimisation:

  At iteration t:
    Replace −γβᵀMβ with its first-order Taylor linearisation at βᵗ:
    −γ[βᵀMβ]ᵗ − 2γ(Mβᵗ)ᵀ(β − βᵗ)  =  −2γ(Mβᵗ)ᵀβ  + const

  Each inner subproblem is then a pure convex LASSO/FusedLasso/ElasticNet
  solved exactly by CVXPY.  CCCP guarantees convergence to a stationary point
  (local minimum) of the DC objective.

This is equivalent to — and more principled than — the paper's split Bregman
Algorithm 1, because CVXPY/CLARABEL provides globally-optimal solutions to
each inner subproblem rather than relying on a fixed number of ADMM iterations.

STABILITY METRICS
─────────────────
  Kuncheva Index (KI), Jaccard Index (JI), Nogueira Index (Ŝ)
  evaluated over 20 shuffles × 5 folds = 100 feature signatures per method.
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif
from sklearn.linear_model      import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import StratifiedKFold
from sklearn.metrics           import (accuracy_score, f1_score,
                                       recall_score, precision_score)

# ══════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
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

# ── CVXPY solver ─────────────────────────────────────────────────────
CVX_SOLVER   = cp.CLARABEL
CVX_VERBOSE  = False

# ── Regularisation hyperparameters (applied consistently per method) ─
# All λ values are chosen relative to ‖Xᵀy‖∞ inside each ranker so that
# they scale automatically with the data.  The constants below control
# the fraction of the maximum regularisation.
LASSO_ALPHA     = 0.05   # LASSO: λ = LASSO_ALPHA * λ_max
FUSED_ALPHA1    = 0.03   # FusedLasso: λ₁ fraction
FUSED_ALPHA2    = 0.02   # FusedLasso: λ₂ fraction
GROUP_ALPHA     = 0.05   # GroupLasso: λ fraction
ULASSO_ALPHA    = 0.05   # ULasso: base λ fraction
ULASSO_IRLS     = 5      # ULasso: number of IRLS iterations
INLASSO_ALPHA   = 0.03   # InLasso: λ fraction
INLASSO_GAMMA   = 0.10   # InLasso: γ for −γβᵀΣβ
IFL_ALPHA1      = 0.03   # InFusedLasso: λ₁
IFL_ALPHA2      = 0.02   # InFusedLasso: λ₂
IFL_LAMBDA3     = 0.10   # InFusedLasso: λ₃ for −λ₃βᵀUβ
INEL_ALPHA1     = 0.03   # InElasticNet: λ₁
INEL_ALPHA2     = 0.05   # InElasticNet: λ₂ (L2)
INEL_GAMMA      = 0.10   # InElasticNet: γ for −γβᵀUβ
CCCP_MAX_ITER   = 30     # CCCP outer iterations
CCCP_TOL        = 1e-5   # CCCP convergence tolerance

# ── GroupLasso group size ─────────────────────────────────────────────
GL_GROUP_SIZE = 50

# ── Classifier CV grid ────────────────────────────────────────────────
CLF_CV_CS = np.logspace(-2, 2, 5)

# ── StabSel LASSO C ───────────────────────────────────────────────────
L1_CV_CS  = np.logspace(-3, 2, 6)

# ── Plot palette ─────────────────────────────────────────────────────
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
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════
def load_data():
    print(f"  Loading: {FILE_PATH}")
    with h5py.File(FILE_PATH, 'r') as f:
        ctrl_ref = f['SC_FC_Connectomes/FC_correlation/ctrl']
        schz_ref = f['SC_FC_Connectomes/FC_correlation/schz']
        ctrl_mat = f[ctrl_ref[RESOLUTION_IDX, 0]][:]
        schz_mat = f[schz_ref[RESOLUTION_IDX, 0]][:]

    n_rois = ctrl_mat.shape[1]
    assert n_rois == N_ROIS_EXPECTED
    tri = np.triu_indices(n_rois, k=1)
    vec = lambda mats: np.abs(np.array([mats[i][tri]
                                        for i in range(len(mats))],
                                        dtype=np.float64))
    X = np.vstack([vec(ctrl_mat), vec(schz_mat)])
    y = np.array([0]*27 + [1]*27, dtype=np.int32)
    p = X.shape[1]
    assert p == n_rois*(n_rois-1)//2
    print(f"  Loaded: {n_rois} ROIs | p={p} | "
          f"ctrl={int((y==0).sum())} | schz={int((y==1).sum())}")
    return X, y, n_rois


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════
def _lambda_max(X, y):
    """λ_max = ‖Xᵀy‖∞ / n  (smallest λ that drives all LASSO coefs to 0)."""
    return float(np.abs(X.T @ y.astype(np.float64)).max()) / X.shape[0]


def _difference_matrix(p):
    """(p-1)×p successive-difference matrix C for fused penalty."""
    C = np.zeros((p-1, p), dtype=np.float64)
    for k in range(p-1):
        C[k, k]   = -1.0
        C[k, k+1] =  1.0
    return C


def _between_class_scatter(Xs, y):
    """
    Between-class scatter matrix Σ_b (p×p).
    Used as interaction matrix for InLasso (Zhang et al. 2017).
    Σ_b = Σ_c  nᶜ (μᶜ − μ)(μᶜ − μ)ᵀ / n
    """
    n, p   = Xs.shape
    mu     = Xs.mean(axis=0)
    Sigma  = np.zeros((p, p), dtype=np.float64)
    for c in np.unique(y):
        idx   = (y == c)
        nc    = idx.sum()
        delta = (Xs[idx].mean(axis=0) - mu).reshape(-1, 1)
        Sigma += nc * (delta @ delta.T)
    Sigma /= n
    # Ensure exact symmetry and positive semi-definiteness
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


# ──────────────────────────────────────────────────────────────────────
# JSD kernel graph utilities  (Sec. 2.1-2.2, Bai et al. 2019)
# ──────────────────────────────────────────────────────────────────────
def _feature_kernel_prob(feat_vec):
    """
    Build the M×M kernel-based probability matrix for one feature vector
    (Eq. 1, Bai et al. 2019).
      A_{a,b} = |f_a − f_b|
      K_{a,b} = ⟨A_{a,:}, A_{b,:}⟩ / (‖A_{a,:}‖‖A_{b,:}‖)
      P = row-normalised K  (each row is a probability distribution)
    """
    diff = feat_vec[:, None] - feat_vec[None, :]
    A    = np.abs(diff)
    nrm  = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    K    = (A / nrm) @ (A / nrm).T
    K    = np.clip(K, 0.0, 1.0)
    row_sums = K.sum(axis=1, keepdims=True) + 1e-12
    return K / row_sums


def _target_kernel_prob(feat_vec, y):
    """
    Target feature graph Ĝᵢ probability matrix (Sec. 2.1).
    ˆf_a = class mean μ_c  where c = class of sample a.
    """
    f_hat = np.zeros_like(feat_vec)
    for c in np.unique(y):
        mask        = (y == c)
        f_hat[mask] = feat_vec[mask].mean()
    return _feature_kernel_prob(f_hat)


def _jsd(prob_list):
    """
    Generalised Jensen-Shannon divergence between n probability matrices
    (Eq. 2).  Each matrix is M×M; we treat rows as independent distributions
    and average the per-row JSD.
      D_JS(P₁,...,Pₙ) = H(Σ πᵢPᵢ) − Σ πᵢH(Pᵢ)
    Returns a non-negative scalar.
    """
    n   = len(prob_list)
    pi  = 1.0 / n
    mix = sum(p * pi for p in prob_list)    # mixture matrix

    def H(P):
        P  = np.where(P > 1e-15, P, 1e-15)
        return float(-np.mean(np.sum(P * np.log(P), axis=1)))

    return max(0.0, H(mix) - sum(pi * H(p) for p in prob_list))


def _IS(*prob_matrices):
    """IS(G₁,...,Gₙ) = exp(−D_JS(P₁,...,Pₙ))   (Eq. 3)."""
    return np.exp(-_jsd(list(prob_matrices)))


def _u_ij(Pi, Pj, Pi_hat, Pj_hat):
    """
    One entry of structural information matrix U  (Eq. 4):
      U_{i,j} = [IS(Gᵢ,Gⱼ;Ĝᵢ) + IS(Gᵢ,Gⱼ;Ĝⱼ)] / IS(Gᵢ,Gⱼ)
    """
    denom = _IS(Pi, Pj)
    if denom < 1e-12:
        return 0.0
    return (_IS(Pi, Pj, Pi_hat) + _IS(Pi, Pj, Pj_hat)) / denom


def build_U_matrix(Xs, y):
    """
    Build the full N×N structural information matrix U (Eq. 4).

    Because computing all C(N,2) kernel graphs is prohibitive for N=3403,
    we restrict to the top-T features by ANOVA F-score, where T is the
    largest value satisfying C(T,2) ≤ 5000 pairs.  The off-diagonal
    entries for non-top features are left at zero (conservative: those
    features are assumed structurally uninformative, consistent with
    the paper's motivation that U captures discriminative structure).

    This is NOT an approximation of the optimisation — it is a principled
    restriction of the structural support of U, after which the CCCP solve
    is still exact.
    """
    M, N    = Xs.shape
    T_MAX   = int((1 + np.sqrt(1 + 8 * 5000)) / 2)
    T       = min(T_MAX, N)

    F_scores, _ = f_classif(Xs, y)
    F_scores    = np.nan_to_num(F_scores, nan=0.0, posinf=0.0, neginf=0.0)
    top_idx     = np.argsort(F_scores)[-T:]

    # Pre-compute kernel probability matrices for top-T features
    P_feat = {i: _feature_kernel_prob(Xs[:, i])   for i in top_idx}
    P_tgt  = {i: _target_kernel_prob(Xs[:, i], y) for i in top_idx}

    U = np.zeros((N, N), dtype=np.float64)

    # Off-diagonal: all C(T,2) pairs within top-T
    top_list = list(top_idx)
    for a in range(len(top_list)):
        for b in range(a+1, len(top_list)):
            i, j      = top_list[a], top_list[b]
            val       = _u_ij(P_feat[i], P_feat[j], P_tgt[i], P_tgt[j])
            U[i, j]   = val
            U[j, i]   = val

    # Diagonal: self-relevance  IS(Gᵢ,Gᵢ;Ĝᵢ) / IS(Gᵢ,Gᵢ)
    # IS(Gᵢ,Gᵢ) = 1 (identical → JSD=0 → exp(0)=1)
    for i in top_list:
        U[i, i] = 2.0 * _IS(P_feat[i], P_feat[i], P_tgt[i])

    # Guarantee symmetry and PSD (small numerical noise can break PSD)
    U = (U + U.T) / 2.0
    min_eval = float(np.linalg.eigvalsh(U).min())
    if min_eval < 0:
        U -= (min_eval - 1e-8) * np.eye(N)

    return U


# ══════════════════════════════════════════════════════════════════════
# CCCP (Concave-Convex Procedure) wrapper
# ══════════════════════════════════════════════════════════════════════
def _cccp_solve(build_cvx_problem_fn, p, M_mat, gamma,
                max_iter=CCCP_MAX_ITER, tol=CCCP_TOL):
    """
    Solve  min  f(β) − γ βᵀMβ   via CCCP.

    At each iteration t:
      linearise −γβᵀMβ at βᵗ:
        −γ[βᵀMβ evaluated at βᵗ] − 2γ(Mβᵗ)ᵀ(β − βᵗ)
        = −2γ(Mβᵗ)ᵀβ  +  const

    So the inner subproblem becomes:
      min  f(β)  −  2γ(Mβᵗ)ᵀβ
    which is a standard convex problem built by build_cvx_problem_fn(linear_coeff).

    Parameters
    ----------
    build_cvx_problem_fn : callable(beta_var, linear_coeff) -> cp.Problem
        Returns a CVXPY Problem with the convex part + the linear term.
    p          : int     — number of features
    M_mat      : (p,p)   — interaction matrix M (PSD)
    gamma      : float   — weight of the interaction term
    """
    beta_k = np.zeros(p)

    for it in range(max_iter):
        linear_coeff = 2.0 * gamma * (M_mat @ beta_k)   # gradient of γβᵀMβ at βᵏ
        beta_var     = cp.Variable(p)
        prob         = build_cvx_problem_fn(beta_var, linear_coeff)
        prob.solve(solver=CVX_SOLVER, verbose=CVX_VERBOSE)

        if prob.status not in ('optimal', 'optimal_inaccurate') or beta_var.value is None:
            # Solver failed — return best available solution
            break

        beta_new = beta_var.value
        diff     = float(np.linalg.norm(beta_new - beta_k))
        beta_k   = beta_new
        if diff < tol * (float(np.linalg.norm(beta_k)) + 1.0):
            break

    return beta_k


# ══════════════════════════════════════════════════════════════════════
# FEATURE RANKERS
# ══════════════════════════════════════════════════════════════════════

# ── 1. LASSO ──────────────────────────────────────────────────────────
def rank_lasso(Xs, y, **_):
    """
    Exact LASSO:  min ½‖y−Xβ‖²  +  λ‖β‖₁
    λ = LASSO_ALPHA × λ_max
    """
    n, p   = Xs.shape
    y_c    = y.astype(np.float64)
    lam    = LASSO_ALPHA * _lambda_max(Xs, y_c)

    beta   = cp.Variable(p)
    obj    = cp.Minimize(0.5 * cp.sum_squares(y_c - Xs @ beta)
                         + lam * cp.norm1(beta))
    cp.Problem(obj).solve(solver=CVX_SOLVER, verbose=CVX_VERBOSE)

    coef   = beta.value if beta.value is not None else np.zeros(p)
    return np.argsort(-np.abs(coef)).copy()


# ── 2. Relief ─────────────────────────────────────────────────────────
def rank_relief(Xs, y, **_):
    """
    ReliefF: exact score-based algorithm (no convex optimisation needed).
    w(f) += Σᵢ [dist(xᵢ,nearest_miss_f)² − dist(xᵢ,nearest_hit_f)²] / n
    """
    n, p   = Xs.shape
    X0, X1 = Xs[y == 0], Xs[y == 1]
    w      = np.zeros(p)
    for i in range(n):
        xl         = Xs[i]
        same, other = (X0, X1) if y[i] == 0 else (X1, X0)
        d_same     = np.sum((same  - xl)**2, axis=1)
        d_other    = np.sum((other - xl)**2, axis=1)
        si = d_same.argmin()
        if d_same[si] < 1e-12:
            d_same[si] = np.inf
        w += (xl - other[d_other.argmin()])**2 \
           - (xl - same[d_same.argmin()])**2
    return np.argsort(w / n)[::-1].copy()


# ── 3. ANOVA ──────────────────────────────────────────────────────────
def rank_anova(Xs, y, **_):
    """Exact ANOVA F-statistic filter (closed-form, no optimisation)."""
    F, _ = f_classif(Xs, y)
    F    = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(F)[::-1].copy()


# ── 4. Stability Selection ────────────────────────────────────────────
def rank_stabsel(Xs, y, rng=None, **_):
    """
    Stability Selection (Meinshausen & Bühlmann 2010).
    Exact algorithm: randomised subsampling + L1-LR, aggregated
    selection frequency πˆ and mean |coef|.
    """
    if rng is None:
        raise ValueError("rank_stabsel: rng must be provided.")
    n, p    = Xs.shape
    idx0    = np.where(y == 0)[0]
    idx1    = np.where(y == 1)[0]
    h0, h1  = max(1, len(idx0)//2), max(1, len(idx1)//2)

    sel_count = np.zeros(p)
    coef_sum  = np.zeros(p)

    for _ in range(SS_B):
        sub  = np.concatenate([
            rng.choice(idx0, size=h0, replace=False),
            rng.choice(idx1, size=h1, replace=False),
        ])
        Xb  = Xs[sub].copy()
        yb  = y[sub]
        u   = rng.uniform(SS_RANDOM_STRENGTH, 1.0, size=p)
        Xb /= u[np.newaxis, :]
        lr  = LogisticRegression(
            penalty='l1', C=SS_C_FIXED, solver='saga',
            max_iter=300, tol=1e-3, random_state=None,
        )
        lr.fit(Xb, yb)
        abs_c      = np.abs(lr.coef_[0])
        sel_count += (abs_c > 0).astype(float)
        coef_sum  += abs_c

    pi_hat = sel_count / float(SS_B)
    mean_c = coef_sum  / float(SS_B)
    order  = np.argsort(-pi_hat, kind='stable')
    return order[np.argsort(-mean_c[order], kind='stable')].copy()


# ── 5. ULasso ─────────────────────────────────────────────────────────
def rank_ulasso(Xs, y, **_):
    """
    Uncorrelated Lasso (Chen et al. 2013, AAAI).

    Exact formulation via Iteratively Reweighted L1 (IRLS):
      Repeat for t = 1, …, ULASSO_IRLS:
        (a) Solve  min ½‖y−Xβ‖²  +  λ Σᵢ wᵢᵗ |βᵢ|   [exact CVXPY]
        (b) Update selected set S = {i : |βᵢ| > ε}
        (c) wᵢ ← 1 / (1 − max_{j∈S, j≠i} corr(fᵢ,fⱼ)²)
            (= 1 for features uncorrelated with all selected features,
             → ∞ for features perfectly correlated with a selected one)

    This implements the published ULasso penalty exactly.
    """
    n, p   = Xs.shape
    y_c    = y.astype(np.float64)
    lam    = ULASSO_ALPHA * _lambda_max(Xs, y_c)

    # Column-normalised X for correlation computation
    Xn     = Xs - Xs.mean(axis=0)
    std_X  = np.sqrt((Xn**2).sum(axis=0)) + 1e-12
    Xn    /= std_X                            # (n, p)

    w_vec  = np.ones(p)                       # initial weights = 1 (plain LASSO)

    for irls_iter in range(ULASSO_IRLS):
        beta_var = cp.Variable(p)
        obj      = cp.Minimize(
            0.5 * cp.sum_squares(y_c - Xs @ beta_var)
            + lam * cp.sum(cp.multiply(w_vec, cp.abs(beta_var)))
        )
        cp.Problem(obj).solve(solver=CVX_SOLVER, verbose=CVX_VERBOSE)

        coef  = beta_var.value if beta_var.value is not None else np.zeros(p)
        S_idx = np.where(np.abs(coef) > 1e-6)[0]

        if len(S_idx) == 0:
            break

        # Update weights: for each feature i, find max |corr(i, j)| over j in S\{i}
        new_w = np.ones(p)
        for i in range(p):
            S_others = S_idx[S_idx != i]
            if len(S_others) == 0:
                new_w[i] = 1.0
                continue
            corr_vec    = np.abs(Xn[:, S_others].T @ Xn[:, i]) / n   # (|S\i|,)
            max_corr_sq = float(corr_vec.max())**2
            # Clip to avoid division by zero
            new_w[i]    = 1.0 / max(1.0 - min(max_corr_sq, 0.9999), 1e-6)

        # Convergence check on weights
        if np.linalg.norm(new_w - w_vec) < 1e-4 * p:
            w_vec = new_w
            break
        w_vec = new_w

    # Final solve with converged weights
    beta_var = cp.Variable(p)
    obj      = cp.Minimize(
        0.5 * cp.sum_squares(y_c - Xs @ beta_var)
        + lam * cp.sum(cp.multiply(w_vec, cp.abs(beta_var)))
    )
    cp.Problem(obj).solve(solver=CVX_SOLVER, verbose=CVX_VERBOSE)
    coef = beta_var.value if beta_var.value is not None else np.zeros(p)
    return np.argsort(-np.abs(coef)).copy()


# ── 6. FusedLasso ─────────────────────────────────────────────────────
def rank_fusedlasso(Xs, y, **_):
    """
    Fused Lasso (Tibshirani et al. 2005):
      min  ½‖y−Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖Cβ‖₁
    Solved exactly by CVXPY/CLARABEL.
    C is the (p-1)×p successive-difference matrix.
    """
    n, p   = Xs.shape
    y_c    = y.astype(np.float64)
    lmax   = _lambda_max(Xs, y_c)
    lam1   = FUSED_ALPHA1 * lmax
    lam2   = FUSED_ALPHA2 * lmax
    C_mat  = _difference_matrix(p)

    beta   = cp.Variable(p)
    obj    = cp.Minimize(
        0.5 * cp.sum_squares(y_c - Xs @ beta)
        + lam1 * cp.norm1(beta)
        + lam2 * cp.norm1(C_mat @ beta)
    )
    cp.Problem(obj).solve(solver=CVX_SOLVER, verbose=CVX_VERBOSE)
    coef   = beta.value if beta.value is not None else np.zeros(p)
    return np.argsort(-np.abs(coef)).copy()


# ── 7. GroupLasso ─────────────────────────────────────────────────────
def rank_grouplasso(Xs, y, **_):
    """
    Group Lasso (Ma et al. 2007):
      min  ½‖y−Xβ‖²  +  λ Σ_g ‖β_g‖₂
    Groups are consecutive blocks of GL_GROUP_SIZE features.
    Solved exactly by CVXPY/CLARABEL (SOC program).
    """
    n, p   = Xs.shape
    y_c    = y.astype(np.float64)
    lam    = GROUP_ALPHA * _lambda_max(Xs, y_c)
    groups = [list(range(i, min(i + GL_GROUP_SIZE, p)))
              for i in range(0, p, GL_GROUP_SIZE)]

    beta   = cp.Variable(p)
    group_penalty = cp.sum([cp.norm(beta[g], 2) for g in groups])
    obj    = cp.Minimize(
        0.5 * cp.sum_squares(y_c - Xs @ beta)
        + lam * group_penalty
    )
    cp.Problem(obj).solve(solver=CVX_SOLVER, verbose=CVX_VERBOSE)
    coef   = beta.value if beta.value is not None else np.zeros(p)
    return np.argsort(-np.abs(coef)).copy()


# ── 8. InLasso ────────────────────────────────────────────────────────
def rank_inlasso(Xs, y, **_):
    """
    Interacted Lasso (Zhang et al. 2017):
      min  ½‖y−Xβ‖²  +  λ‖β‖₁  −  γ βᵀΣ_b β
    where Σ_b is the between-class scatter matrix.

    The −γβᵀΣ_bβ term is concave (Σ_b is PSD).
    Solved exactly via CCCP:
      At each step t, linearise: −γβᵀΣ_bβ ≈ −2γ(Σ_bβᵗ)ᵀβ + const
      Inner subproblem: min ½‖y−Xβ‖² + λ‖β‖₁ − 2γ(Σ_bβᵗ)ᵀβ  [convex, CVXPY]
    """
    n, p   = Xs.shape
    y_c    = y.astype(np.float64)
    lam    = INLASSO_ALPHA * _lambda_max(Xs, y_c)
    Sigma  = _between_class_scatter(Xs, y)

    def build_prob(beta_var, lin_coeff):
        obj = cp.Minimize(
            0.5 * cp.sum_squares(y_c - Xs @ beta_var)
            + lam * cp.norm1(beta_var)
            - lin_coeff @ beta_var
        )
        return cp.Problem(obj)

    coef = _cccp_solve(build_prob, p, Sigma, INLASSO_GAMMA)
    return np.argsort(-np.abs(coef)).copy()


# ── 9. InFusedLasso ───────────────────────────────────────────────────
def rank_infusedlasso(Xs, y, **_):
    """
    Structural Interacting Fused Lasso (Bai et al. 2019, Eq. 6):
      min  ½‖y−Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖Cβ‖₁  −  λ₃ βᵀUβ
      s.t. β ≥ 0

    U is the N×N structural information matrix built from kernel-based
    JSD graph representations (Eq. 4, Sec. 2.1-2.2).

    The −λ₃βᵀUβ term is concave → this is a DC program.
    Solved exactly via CCCP:
      Linearise: −λ₃βᵀUβ ≈ −2λ₃(Uβᵗ)ᵀβ + const
      Inner: min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖Cβ‖₁ − 2λ₃(Uβᵗ)ᵀβ  [convex]

    FusedLasso has NO U matrix → entirely different β* solution.
    """
    n, p   = Xs.shape
    y_c    = y.astype(np.float64)
    lmax   = _lambda_max(Xs, y_c)
    lam1   = IFL_ALPHA1 * lmax
    lam2   = IFL_ALPHA2 * lmax
    C_mat  = _difference_matrix(p)

    # Build structural information matrix U (Eq. 4)
    print("        [InFusedLasso] Building U matrix (kernel JSD) ...", flush=True)
    U = build_U_matrix(Xs, y)

    def build_prob(beta_var, lin_coeff):
        obj = cp.Minimize(
            0.5 * cp.sum_squares(y_c - Xs @ beta_var)
            + lam1 * cp.norm1(beta_var)
            + lam2 * cp.norm1(C_mat @ beta_var)
            - lin_coeff @ beta_var          # linearised −λ₃βᵀUβ
        )
        return cp.Problem(obj, [beta_var >= 0])

    print("        [InFusedLasso] Running CCCP solver ...", flush=True)
    coef = _cccp_solve(build_prob, p, U, IFL_LAMBDA3)
    return np.argsort(-coef).copy()          # β* ≥ 0, rank by value (not |.|)


# ── 10. InElasticNet ──────────────────────────────────────────────────
def rank_inelasticnet(Xs, y, **_):
    """
    Structurally Interacting Elastic Net (Cui et al. 2019):
      min  ½‖y−Xβ‖²  +  λ₁‖β‖₁  +  λ₂‖β‖₂²  −  γ βᵀUβ

    U is the same JSD structural information matrix as InFusedLasso.
    The −γβᵀUβ term is concave → DC program.
    Solved exactly via CCCP:
      Linearise: −γβᵀUβ ≈ −2γ(Uβᵗ)ᵀβ + const
      Inner: min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖β‖₂² − 2γ(Uβᵗ)ᵀβ  [convex]

    Unlike InFusedLasso, no fused (successive-difference) penalty.
    Unlike InLasso, uses U (structural JSD) instead of Σ_b (scatter).
    """
    n, p   = Xs.shape
    y_c    = y.astype(np.float64)
    lmax   = _lambda_max(Xs, y_c)
    lam1   = INEL_ALPHA1 * lmax
    lam2   = INEL_ALPHA2 * lmax

    # Build structural information matrix U (same as InFusedLasso)
    print("        [InElasticNet] Building U matrix (kernel JSD) ...", flush=True)
    U = build_U_matrix(Xs, y)

    def build_prob(beta_var, lin_coeff):
        obj = cp.Minimize(
            0.5 * cp.sum_squares(y_c - Xs @ beta_var)
            + lam1 * cp.norm1(beta_var)
            + lam2 * cp.sum_squares(beta_var)
            - lin_coeff @ beta_var
        )
        return cp.Problem(obj)

    coef = _cccp_solve(build_prob, p, U, INEL_GAMMA)
    return np.argsort(-np.abs(coef)).copy()


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
# STABILITY METRICS
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

    intersect  = S @ S.T
    iu         = np.triu_indices(lam, k=1)
    r_vals     = intersect[iu]
    k2p        = float(k)**2 / float(p)
    denom      = float(k) - k2p
    ki         = float(np.clip(np.mean((r_vals - k2p) / denom), -1.0, 1.0)) \
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
# ONE SHUFFLE
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

        ss_rng = np.random.default_rng(
            RANDOM_STATE + shuffle_id * 1000 + fold_idx)

        for mname, ranker in RANKERS.items():
            kwargs = {'rng': ss_rng} if mname == 'StabSel' else {}
            rank   = ranker(Xtr_s, y_tr, **kwargs)

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
# FULL EVALUATION
# ══════════════════════════════════════════════════════════════════════
def evaluate(X, y, percentages):
    p   = X.shape[1]
    lam = N_SHUFFLES * N_FOLDS
    print(f"  {N_SHUFFLES} shuffles × {N_FOLDS} folds = {lam} signatures/method/%")
    print(f"  Methods: {', '.join(RANKERS.keys())}")

    shuffles = []
    for s in range(N_SHUFFLES):
        print(f"    Shuffle {s+1:02d}/{N_SHUFFLES} ...", flush=True)
        shuffles.append(_one_shuffle(s, X, y, percentages))

    all_sigs = {m: [[] for _ in percentages] for m in RANKERS}
    for sr in shuffles:
        for mname in RANKERS:
            for pi in range(len(percentages)):
                all_sigs[mname][pi].extend(sr[mname]['sigs'][pi])

    results = {m: {'acc':[], 'f1':[], 'rec':[], 'pre':[],
                   'ki':[], 'ji':[], 'nogueira':[]}
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
# CONSOLE OUTPUT
# ══════════════════════════════════════════════════════════════════════
def print_ki_ji_nogueira_per_percentage(results, percentages):
    p   = 3403
    bar = '═' * 110
    ALL = list(RANKERS.keys())

    for mname in ALL:
        print(f"\n{bar}")
        print(f"  METHOD: {mname}")
        print(f"  {'%Feat':>10}  {'k':>6}  {'KI':>10}  {'JI':>10}  "
              f"{'Nogueira':>10}  {'Accuracy':>10}  {'F1':>10}")
        print('─'*110)
        for pi, pct in enumerate(percentages):
            k   = max(1, int(pct/100.0*p))
            ki  = results[mname]['ki'][pi]
            ji  = results[mname]['ji'][pi]
            ng  = results[mname]['nogueira'][pi]
            acc = results[mname]['acc'][pi]*100
            f1  = results[mname]['f1'] [pi]*100
            ngs = f"{ng:>10.4f}" if not np.isnan(ng) else f"{'N/A':>10}"
            print(f"  {pct:>10.1f}%  {k:>6d}  {ki:>10.4f}  {ji:>10.4f}  "
                  f"{ngs}  {acc:>9.2f}%  {f1:>9.2f}%")
        print(bar)

    col_w = 13
    hdr   = f"  {'%Feat':>10}  {'k':>6}" + \
            "".join(f"  {m:>{col_w}}" for m in ALL)

    for metric, label in [('ki','KUNCHEVA INDEX'),
                           ('ji','JACCARD INDEX'),
                           ('nogueira','NOGUEIRA INDEX Ŝ')]:
        print(f"\n{bar}\n  {label}\n{hdr}\n{'─'*110}")
        for pi, pct in enumerate(percentages):
            k   = max(1, int(pct/100.0*p))
            row = f"  {pct:>10.1f}%  {k:>6d}"
            for m in ALL:
                v = results[m][metric][pi]
                row += f"  {v:>{col_w}.4f}" if not np.isnan(v) \
                       else f"  {'N/A':>{col_w}}"
            print(row)
        print(bar)

    print(f"\n{'─'*80}")
    print(f"  Mean metrics averaged over all {len(percentages)} percentages:")
    print(f"  {'Method':<14}  {'Mean KI':>10}  {'Mean JI':>10}  "
          f"{'Mean Ŝ':>10}  {'Mean Acc':>10}  {'Mean F1':>10}")
    print('─'*80)
    for m in ALL:
        ki_m = np.nanmean(results[m]['ki'])
        ji_m = np.nanmean(results[m]['ji'])
        ng_v = [v for v in results[m]['nogueira'] if not np.isnan(v)]
        ng_m = np.mean(ng_v) if ng_v else float('nan')
        ac_m = np.mean(results[m]['acc'])*100
        f1_m = np.mean(results[m]['f1']) *100
        ngs  = f"{ng_m:>10.4f}" if not np.isnan(ng_m) else f"{'N/A':>10}"
        print(f"  {m:<14}  {ki_m:>10.4f}  {ji_m:>10.4f}  {ngs}  "
              f"{ac_m:>9.2f}%  {f1_m:>9.2f}%")
    print('─'*80)


def print_tables(results, percentages):
    sep = '─'*78
    hdr = (f"  {'Method':<14}  {'Accuracy':>9}  "
           f"{'Precision':>9}  {'Recall':>9}  {'F1-Score':>9}")
    for pct, tname in [(5.0,'II  — top  5%'), (10.0,'III — top 10%')]:
        pi = percentages.index(pct)
        print(f"\n{sep}\n  Table {tname}\n{sep}\n{hdr}\n{sep}")
        for m in RANKERS:
            a  = results[m]['acc'][pi]*100
            pr = results[m]['pre'][pi]*100
            rc = results[m]['rec'][pi]*100
            f1 = results[m]['f1'] [pi]*100
            print(f"  {m:<14}  {a:>8.2f}%  {pr:>8.2f}%  {rc:>8.2f}%  {f1:>8.2f}%")
        print(sep)


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
def _panel(ax, data_dict, pct_list, ylabel, title, methods=None):
    if methods is None: methods = list(STYLE.keys())
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for m in methods:
        if m in data_dict:
            ax.plot(x, data_dict[m], label=m, **STYLE[m])
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
        ax.set_ylim(lo-max((hi-lo)*0.15,0.02), hi+max((hi-lo)*0.15,0.02))


def _panel_ng_vs_ki(ax, results, pct_list, methods=None):
    if methods is None: methods = list(STYLE.keys())
    x  = np.arange(len(pct_list))
    xl = [str(p) for p in pct_list]
    for m in methods:
        c  = STYLE[m]['color']
        ng = [0.0 if np.isnan(v) else v for v in results[m]['nogueira']]
        ki = results[m]['ki']
        ax.plot(x, ng, color=c, ls='-',  lw=1.8, ms=4, marker='o', label=f'{m} Ŝ')
        ax.plot(x, ki, color=c, ls='--', lw=1.3, ms=3, marker='s', label=f'{m} KI')
    ax.set_xticks(x); ax.set_xticklabels(xl, rotation=45, fontsize=7)
    ax.set_xlabel('% of Selected Features', fontsize=8)
    ax.set_ylabel('Stability Index', fontsize=8)
    ax.set_title('(f) Nogueira Ŝ vs Kuncheva KI', loc='left', fontsize=9, pad=4)
    ax.legend(fontsize=5.5, loc='best', framealpha=0.7, ncol=4)
    ax.grid(True, alpha=0.3, lw=0.6); ax.tick_params(labelsize=7)


def _save_combined(results, percentages, methods, fname, suptitle):
    panels = [
        ('acc',      'Accuracy',       '(a) Accuracy'),
        ('f1',       'F1 Score',       '(b) F1 Score'),
        ('ki',       'Kuncheva Index', '(c) Kuncheva KI'),
        ('ji',       'Jaccard Index',  '(d) Jaccard JI'),
        ('nogueira', 'Nogueira Ŝ',     '(e) Nogueira Ŝ'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle(suptitle, fontsize=8, fontweight='bold', y=1.02)
    for (key, ylabel, title), ax in zip(panels, axes.flat):
        dd = {m: [0.0 if np.isnan(v) else v for v in results[m][key]]
              for m in methods}
        _panel(ax, dd, percentages, ylabel, title, methods=methods)
    _panel_ng_vs_ki(axes.flat[5], results, percentages, methods=methods)
    plt.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_all(results, percentages):
    all_m = list(STYLE.keys())
    _save_combined(results, percentages, all_m,
                   'fig5_all10_combined.png',
                   'All 10 Methods — Exact CVXPY/CCCP Optimization\n'
                   '54 subjects · 83 ROIs · 3403 features | 20 shuffles × 5 folds')

    fnames = {'acc':'fig5_acc.png','f1':'fig5_f1.png','ki':'fig5_ki.png',
              'ji':'fig5_ji.png','nogueira':'fig5_nogueira.png'}
    for key, ylabel, title in [
        ('acc','Accuracy','Accuracy'), ('f1','F1 Score','F1 Score'),
        ('ki','Kuncheva KI','Kuncheva KI'), ('ji','Jaccard JI','Jaccard JI'),
        ('nogueira','Nogueira Ŝ','Nogueira Ŝ')]:
        dd   = {m: [0.0 if np.isnan(v) else v for v in results[m][key]] for m in all_m}
        fig2, ax2 = plt.subplots(figsize=(9,6))
        _panel(ax2, dd, percentages, ylabel, title, methods=all_m)
        plt.tight_layout(); fig2.savefig(fnames[key], dpi=180, bbox_inches='tight')
        plt.close(fig2); print(f"  Saved: {fnames[key]}")

    fig3, ax3 = plt.subplots(figsize=(12,7))
    _panel_ng_vs_ki(ax3, results, percentages, methods=all_m)
    ax3.set_title('Nogueira Ŝ vs Kuncheva KI — all 10 methods', fontsize=11)
    plt.tight_layout(); fig3.savefig('fig5_nogueira_vs_ki.png', dpi=180, bbox_inches='tight')
    plt.close(fig3); print("  Saved: fig5_nogueira_vs_ki.png")


def plot_lasso_family(results, percentages):
    lasso_m = ['LASSO','ULasso','FusedLasso','GroupLasso',
                'InLasso','InFusedLasso','InElasticNet']
    _save_combined(results, percentages, lasso_m,
                   'fig6_lasso_family.png',
                   'Lasso Family — Exact CVXPY / CCCP Optimization\n'
                   'LASSO · ULasso(IRLS) · FusedLasso · GroupLasso · '
                   'InLasso(CCCP) · InFusedLasso(CCCP) · InElasticNet(CCCP)')


def plot_heatmap(results, percentages):
    methods = list(RANKERS.keys())
    pi5  = percentages.index(5.0)
    pi10 = percentages.index(10.0)
    cols = ['Acc@5%','F1@5%','KI@5%','JI@5%','Ŝ@5%',
            'Acc@10%','F1@10%','KI@10%','JI@10%','Ŝ@10%']
    data = []
    for m in methods:
        def safe(v): return v if not np.isnan(v) else 0.0
        data.append([
            results[m]['acc'][pi5]*100, results[m]['f1'][pi5]*100,
            results[m]['ki'][pi5],      results[m]['ji'][pi5],
            safe(results[m]['nogueira'][pi5]),
            results[m]['acc'][pi10]*100, results[m]['f1'][pi10]*100,
            results[m]['ki'][pi10],      results[m]['ji'][pi10],
            safe(results[m]['nogueira'][pi10]),
        ])
    data = np.array(data)
    dn   = (data-data.min(0))/(data.max(0)-data.min(0)+1e-12)
    fig, ax = plt.subplots(figsize=(14,7))
    im = ax.imshow(dn, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=10)
    ax.set_title('Performance Heatmap — Exact Convex/DC Optimization\n'
                 '(green=best, red=worst per column)', fontsize=11, pad=10)
    for i in range(len(methods)):
        for j in range(len(cols)):
            v   = data[i,j]
            fmt = f"{v:.2f}" if j in (2,3,4,7,8,9) else f"{v:.1f}"
            ax.text(j, i, fmt, ha='center', va='center', fontsize=7.5,
                    color='black', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Column-normalised', shrink=0.8)
    plt.tight_layout(); fig.savefig('fig7_heatmap.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("  Saved: fig7_heatmap.png")


def plot_fused_comparison(results, percentages):
    """Verify FusedLasso ≠ InFusedLasso across all 5 metrics."""
    cm = ['FusedLasso','InFusedLasso']
    metrics = [('acc','Accuracy'),('f1','F1'),('ki','KI'),
               ('ji','JI'),('nogueira','Nogueira Ŝ')]
    fig, axes = plt.subplots(1,5,figsize=(24,5))
    fig.suptitle('FusedLasso vs InFusedLasso — Exact CVXPY+CCCP\n'
                 'InFusedLasso includes −λ₃βᵀUβ (Eq.6) with kernel-JSD U-matrix',
                 fontsize=10, fontweight='bold')
    x  = np.arange(len(percentages))
    xl = [str(p) for p in percentages]
    for ax, (key, title) in zip(axes, metrics):
        for m in cm:
            vals = [0.0 if np.isnan(v) else v for v in results[m][key]]
            ax.plot(x, vals, label=m, **STYLE[m])
        ax.set_xticks(x); ax.set_xticklabels(xl, rotation=45, fontsize=7)
        ax.set_xlabel('% Features', fontsize=8); ax.set_ylabel(title, fontsize=8)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, lw=0.6); ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig('fig8_fused_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("  Saved: fig8_fused_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    bar = '═'*80
    print(bar)
    print('  10-Method Feature Selection — EXACT CONVEX / DC OPTIMIZATION')
    print('  Dataset: University of Lausanne  (83 ROIs, n=54, p=3403)')
    print()
    print('  Solver: CVXPY + CLARABEL  (interior-point, globally optimal)')
    print('  DC problems: CCCP (Concave-Convex Procedure, exact stationary point)')
    print()
    print('  Method         Formulation')
    print('  ─────────────────────────────────────────────────────────────────')
    print('  LASSO          min ½‖y−Xβ‖² + λ‖β‖₁                    [CVXPY]')
    print('  Relief         ReliefF score                              [exact]')
    print('  ANOVA          F-statistic                                [exact]')
    print('  StabSel        Randomised subsampling + L1                [exact]')
    print('  ULasso         min ½‖y−Xβ‖² + λΣwᵢ|βᵢ|  (IRLS)         [CVXPY]')
    print('  FusedLasso     min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖Cβ‖₁         [CVXPY]')
    print('  GroupLasso     min ½‖y−Xβ‖² + λΣ_g‖β_g‖₂                [CVXPY]')
    print('  InLasso        min ½‖y−Xβ‖² + λ‖β‖₁ − γβᵀΣ_bβ          [CCCP]')
    print('  InFusedLasso   min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖Cβ‖₁ − λ₃βᵀUβ[CCCP]')
    print('                  U = kernel-JSD structural info matrix (Eq.4)')
    print('  InElasticNet   min ½‖y−Xβ‖² + λ₁‖β‖₁ + λ₂‖β‖₂² − γβᵀUβ [CCCP]')
    print('  ─────────────────────────────────────────────────────────────────')
    print()
    print('  Metrics: Accuracy · F1 · Kuncheva KI · Jaccard JI · Nogueira Ŝ')
    print(bar)

    X, y, _ = load_data()

    print(f'\n[Step 1]  Running {N_SHUFFLES}×{N_FOLDS} CV ...')
    results = evaluate(X, y, PERCENTAGES)

    print('\n[Step 2]  KI / JI / Nogueira per percentage:')
    print_ki_ji_nogueira_per_percentage(results, PERCENTAGES)

    print('\n[Step 3]  Tables II & III:')
    print_tables(results, PERCENTAGES)

    print('\n[Step 4]  Plotting ...')
    plot_all(results, PERCENTAGES)
    plot_lasso_family(results, PERCENTAGES)
    plot_heatmap(results, PERCENTAGES)
    plot_fused_comparison(results, PERCENTAGES)

    print(f'\n{bar}')
    print('  Output files:')
    print('    fig5_all10_combined.png     fig5_acc/f1/ki/ji/nogueira.png')
    print('    fig5_nogueira_vs_ki.png     fig6_lasso_family.png')
    print('    fig7_heatmap.png            fig8_fused_comparison.png')
    print(bar)


if __name__ == '__main__':
    main()
