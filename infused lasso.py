"""
STEP 1 / 10 — InFusedLasso Feature Selection
=============================================
Formulation (Bai et al., NeurIPS 2019, Eq. 6):

    min  1/2||y - Xb||^2  +  l1||b||_1  +  l2||Cb||_1  -  l3 b'Ub   s.t. b >= 0

Solver    : CCCP  (linearise concave term; inner subproblem -> CVXPY/CLARABEL)
Parameters: (l1, l2, l3) selected ENTIRELY by inner CV - no hand-picking.
            Grid anchored to data-driven lmax = ||X'(y-ybar)||_inf / n.

Key optimisation for speed (no leakage):
  U matrix precomputed ONCE per inner-fold training set.
  All 27 lambda combos for that inner fold reuse the cached U.
  CCCP uses 3 iterations for CV search, 30 for the final fit.

Outputs per feature-percentage:
  Accuracy, F1, Precision, Recall, Kuncheva KI, Jaccard JI, Nogueira S-hat
"""

import warnings, time, os, json
warnings.filterwarnings('ignore')

import numpy as np
import cvxpy as cp

from sklearn.feature_selection  import f_classif
from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import StratifiedKFold, KFold
from sklearn.linear_model       import LogisticRegressionCV
from sklearn.metrics            import (accuracy_score, f1_score,
                                        recall_score, precision_score)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
FILE_PATH       = '/kaggle/input/datasets/kinnarhalder/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat'
RESOLUTION_IDX  = 0
N_ROIS_EXPECTED = 83

PERCENTAGES  = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 60.0, 70.0, 80.0]
N_SHUFFLES   = 2      # set to 20 for publication-quality results
N_FOLDS      = 5
RANDOM_STATE = 42

INNER_CV_FOLDS  = 3   # inner folds for lambda CV
N_LAM_3D        = 3   # 3^3 = 27 lambda combos
CCCP_ITER_CV    = 3   # fast CCCP during CV grid search
CCCP_ITER_FINAL = 30  # full CCCP for final model
CCCP_TOL        = 1e-5

CLF_CV_CS   = np.logspace(-2, 2, 5)
CVX_SOLVER  = cp.CLARABEL
CVX_VERBOSE = False


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
def load_data():
    if os.path.exists(FILE_PATH):
        print(f"  Loading: {FILE_PATH}")
        import h5py
        with h5py.File(FILE_PATH, 'r') as f:
            ctrl_ref = f['SC_FC_Connectomes/FC_correlation/ctrl']
            schz_ref = f['SC_FC_Connectomes/FC_correlation/schz']
            ctrl_mat = f[ctrl_ref[RESOLUTION_IDX, 0]][:]
            schz_mat = f[schz_ref[RESOLUTION_IDX, 0]][:]
        n_rois = ctrl_mat.shape[1]
        assert n_rois == N_ROIS_EXPECTED
        tri  = np.triu_indices(n_rois, k=1)
        vec  = lambda mats: np.abs(np.array(
                   [mats[i][tri] for i in range(len(mats))], dtype=np.float64))
        X = np.vstack([vec(ctrl_mat), vec(schz_mat)])
        y = np.array([0]*27 + [1]*27, dtype=np.int32)
    else:
        print("  *** .mat file not found - using SYNTHETIC stand-in dataset ***")
        print("  *** Set FILE_PATH to the real .mat path for actual results  ***\n")
        rng   = np.random.default_rng(RANDOM_STATE)
        n_rois = N_ROIS_EXPECTED
        p     = n_rois * (n_rois - 1) // 2   # 3403 features
        X     = rng.standard_normal((54, p)) * 0.5
        y     = np.array([0]*27 + [1]*27, dtype=np.int32)
        sig   = rng.choice(p, size=30, replace=False)
        X[27:, sig] += 1.2
        X = np.abs(X)

    p = X.shape[1]
    assert p == N_ROIS_EXPECTED * (N_ROIS_EXPECTED - 1) // 2
    print(f"  n={X.shape[0]}, p={p}, ctrl={int((y==0).sum())}, schz={int((y==1).sum())}")
    return X, y, p


# ─────────────────────────────────────────────────────────────────────
# MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────
def _lambda_max(X, y):
    return float(np.abs(X.T @ (y.astype(np.float64) - y.mean())).max()) / X.shape[0]

def _lam_grid(lmax, ratio, n_pts):
    return np.logspace(np.log10(lmax), np.log10(lmax / ratio), n_pts)

def _difference_matrix(p):
    C = np.zeros((p-1, p), dtype=np.float64)
    for k in range(p-1):
        C[k, k] = -1.0; C[k, k+1] = 1.0
    return C


# ─────────────────────────────────────────────────────────────────────
# JSD KERNEL GRAPH  (Bai et al. 2019, Sec 2.1-2.2)
# ─────────────────────────────────────────────────────────────────────
def _feature_kernel_prob(fvec):
    A   = np.abs(fvec[:, None] - fvec[None, :])
    nrm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    K   = np.clip((A / nrm) @ (A / nrm).T, 0.0, 1.0)
    rs  = K.sum(1, keepdims=True) + 1e-12
    return K / rs

def _target_kernel_prob(fvec, y):
    fhat = np.zeros_like(fvec)
    for c in np.unique(y):
        m = (y == c); fhat[m] = fvec[m].mean()
    return _feature_kernel_prob(fhat)

def _jsd(prob_list):
    pi  = 1.0 / len(prob_list)
    mix = sum(pi * p for p in prob_list)
    def H(P):
        P = np.where(P > 1e-15, P, 1e-15)
        return float(-np.mean(np.sum(P * np.log(P), axis=1)))
    return max(0.0, H(mix) - sum(pi * H(p) for p in prob_list))

def _IS(*probs):
    return np.exp(-_jsd(list(probs)))

def _u_ij(Pi, Pj, Phi, Phj):
    d = _IS(Pi, Pj)
    return (_IS(Pi, Pj, Phi) + _IS(Pi, Pj, Phj)) / d if d > 1e-12 else 0.0

def build_U_matrix(X, y):
    """N x N structural information matrix U (Eq. 4, Bai et al. 2019)."""
    M, N    = X.shape
    T       = min(int((1 + np.sqrt(1 + 8*5000)) / 2), N)
    F, _    = f_classif(X, y)
    top_idx = list(np.argsort(np.nan_to_num(F))[-T:])
    Pf = {i: _feature_kernel_prob(X[:, i]) for i in top_idx}
    Pt = {i: _target_kernel_prob(X[:, i], y) for i in top_idx}
    U  = np.zeros((N, N))
    for a in range(len(top_idx)):
        for b in range(a+1, len(top_idx)):
            i, j = top_idx[a], top_idx[b]
            v = _u_ij(Pf[i], Pf[j], Pt[i], Pt[j])
            U[i, j] = v; U[j, i] = v
    for i in top_idx:
        U[i, i] = 2.0 * _IS(Pf[i], Pf[i], Pt[i])
    U = (U + U.T) / 2.0
    ev_min = float(np.linalg.eigvalsh(U).min())
    if ev_min < 0:
        U += (-ev_min + 1e-8) * np.eye(N)
    return U


# ─────────────────────────────────────────────────────────────────────
# CCCP SOLVER
# ─────────────────────────────────────────────────────────────────────
def _cccp(build_prob_fn, p, M, gamma, max_iter=CCCP_ITER_FINAL, tol=CCCP_TOL):
    """
    Minimise f(b) - gamma * b'Mb  via CCCP.
    Concave term linearised at b_k: supergradient = 2*gamma*M*b_k.
    build_prob_fn(b_var, lin) -> cp.Problem for:  min f(b) - lin . b
    """
    beta_k = np.zeros(p)
    for _ in range(max_iter):
        lin  = 2.0 * gamma * (M @ beta_k)
        bv   = cp.Variable(p)
        prob = build_prob_fn(bv, lin)
        prob.solve(solver=CVX_SOLVER, verbose=CVX_VERBOSE)
        if prob.status not in ('optimal', 'optimal_inaccurate') or bv.value is None:
            break
        d      = float(np.linalg.norm(bv.value - beta_k))
        beta_k = bv.value.copy()
        if d < tol * (float(np.linalg.norm(beta_k)) + 1.0):
            break
    return beta_k


# ─────────────────────────────────────────────────────────────────────
# InFusedLasso RANKER  (U cached per inner fold for speed)
# ─────────────────────────────────────────────────────────────────────
def rank_infusedlasso(Xs, y, verbose=True):
    """
    Rank features by InFusedLasso coefficient magnitude.

    Speed trick: U is precomputed once per inner-fold training set.
    All 27 (l1,l2,l3) combos for that fold reuse the same cached U,
    reducing U builds from 27x3=81 down to just 3 per ranker call.
    """
    n, p   = Xs.shape
    y_c    = y.astype(np.float64)
    C_mat  = _difference_matrix(p)

    # Data-driven lambda grids - NO manual choice
    lmax   = _lambda_max(Xs, y_c)
    g1     = _lam_grid(lmax,        300.0,  N_LAM_3D)  # l1 sparsity
    g2     = _lam_grid(lmax / 5.0, 1500.0,  N_LAM_3D)  # l2 fused
    g3     = np.logspace(-2, 0, N_LAM_3D)               # l3 interaction

    if verbose:
        print(f"          lmax={lmax:.4e}  "
              f"g1=[{g1[0]:.2e}..{g1[-1]:.2e}]  "
              f"g2=[{g2[0]:.2e}..{g2[-1]:.2e}]  "
              f"g3=[{g3[0]:.2e}..{g3[-1]:.2e}]")

    kf     = KFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                   random_state=RANDOM_STATE)
    splits = list(kf.split(Xs))

    # Precompute U once per inner fold (key speed optimisation)
    if verbose:
        print(f"          Precomputing U for {INNER_CV_FOLDS} inner folds ...",
              flush=True)
    U_cache = {}
    for fi, (tr, _) in enumerate(splits):
        t0 = time.time()
        U_cache[fi] = build_U_matrix(Xs[tr], y[tr])
        if verbose:
            print(f"            inner fold {fi+1}: U built in {time.time()-t0:.1f}s",
                  flush=True)

    # 3-D CV grid search
    best_l1, best_l2, best_l3, best_cv = g1[0], g2[0], g3[0], np.inf
    n_combos = len(g1) * len(g2) * len(g3)
    done = 0
    if verbose:
        print(f"          3-D CV: {n_combos} combos x {INNER_CV_FOLDS} folds "
              f"(CCCP {CCCP_ITER_CV} iter each) ...", flush=True)

    for lam1 in g1:
        for lam2 in g2:
            for lam3 in g3:
                mse_folds = []
                for fi, (tr, te) in enumerate(splits):
                    Xtr, ytr = Xs[tr], y[tr]
                    Xte, yte = Xs[te], y[te]
                    U_tr     = U_cache[fi]
                    ytrc     = ytr.astype(float)

                    def bp(bv, lin, _X=Xtr, _y=ytrc, _l1=lam1, _l2=lam2):
                        return cp.Problem(
                            cp.Minimize(
                                0.5 * cp.sum_squares(_y - _X @ bv)
                                + _l1 * cp.norm1(bv)
                                + _l2 * cp.norm1(C_mat @ bv)
                                - lin @ bv),
                            [bv >= 0])

                    beta = _cccp(bp, Xtr.shape[1], U_tr, lam3,
                                 max_iter=CCCP_ITER_CV)
                    pred = Xte @ beta
                    mse_folds.append(
                        float(np.mean((yte.astype(float) - pred)**2)))

                cv = float(np.mean(mse_folds))
                if cv < best_cv:
                    best_cv = cv
                    best_l1, best_l2, best_l3 = lam1, lam2, lam3
                done += 1

    if verbose:
        print(f"          CV done. Best l1={best_l1:.3e}  l2={best_l2:.3e}  "
              f"l3={best_l3:.3e}  (CV-MSE={best_cv:.4f})", flush=True)

    # Final fit on full training set with best lambdas
    if verbose:
        print(f"          Final fit (CCCP {CCCP_ITER_FINAL} iter) ...", flush=True)
    t0     = time.time()
    U_full = build_U_matrix(Xs, y)

    def bp_final(bv, lin):
        return cp.Problem(
            cp.Minimize(
                0.5 * cp.sum_squares(y_c - Xs @ bv)
                + best_l1 * cp.norm1(bv)
                + best_l2 * cp.norm1(C_mat @ bv)
                - lin @ bv),
            [bv >= 0])

    coef = _cccp(bp_final, p, U_full, best_l3, max_iter=CCCP_ITER_FINAL)
    if verbose:
        print(f"          Final fit done in {time.time()-t0:.1f}s  "
              f"nonzero={(np.abs(coef)>1e-6).sum()}", flush=True)

    return np.argsort(-coef).copy()   # b >= 0, rank descending by value


# ─────────────────────────────────────────────────────────────────────
# STABILITY METRICS
# ─────────────────────────────────────────────────────────────────────
def compute_ki_ji(signatures, p):
    lam = len(signatures)
    if lam < 2: return np.nan, np.nan
    k = len(signatures[0])
    if k == 0: return np.nan, np.nan
    S = np.zeros((lam, p), dtype=np.float32)
    for i, sig in enumerate(signatures):
        valid = sig[(sig >= 0) & (sig < p)]
        S[i, valid] = 1.0
    isect = S @ S.T
    iu    = np.triu_indices(lam, k=1)
    r     = isect[iu]
    k2p   = float(k)**2 / float(p)
    denom = float(k) - k2p
    ki    = float(np.clip(np.mean((r - k2p) / denom), -1.0, 1.0)) \
            if abs(denom) > 1e-12 else 1.0
    uv    = 2.0 * float(k) - r
    ji    = float(np.mean(np.where(uv > 0, r / uv, 1.0)))
    return ki, ji

def compute_nogueira(signatures, p):
    lam = len(signatures)
    if lam < 2: return np.nan
    Z = np.zeros((lam, p))
    for i, sig in enumerate(signatures):
        valid = sig[(sig >= 0) & (sig < p)]
        Z[i, valid] = 1.0
    k_bar = float(Z.sum(1).mean())
    if k_bar <= 0.0 or k_bar >= float(p): return np.nan
    V_bar = float(np.mean(Z.mean(0) * (1.0 - Z.mean(0))))
    denom = k_bar * (1.0 - k_bar / float(p))
    if abs(denom) < 1e-12: return np.nan
    return float(1.0 - float(p) * V_bar / denom)


# ─────────────────────────────────────────────────────────────────────
# ONE SHUFFLE
# ─────────────────────────────────────────────────────────────────────
def _one_shuffle(shuffle_id, X, y, percentages, p):
    ks  = [max(1, int(pct / 100.0 * p)) for pct in percentages]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE + shuffle_id)
    acc_s = np.zeros(len(percentages))
    f1_s  = np.zeros(len(percentages))
    rec_s = np.zeros(len(percentages))
    pre_s = np.zeros(len(percentages))
    sigs  = [[] for _ in percentages]

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        print(f"      Outer fold {fold_idx+1}/{N_FOLDS}", flush=True)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sc         = StandardScaler()
        Xtr_s      = sc.fit_transform(X_tr)
        Xte_s      = sc.transform(X_te)

        t0   = time.time()
        rank = rank_infusedlasso(Xtr_s, y_tr, verbose=True)
        print(f"      -> rank done in {time.time()-t0:.1f}s", flush=True)

        for pi, k in enumerate(ks):
            sel = rank[:k]
            clf = LogisticRegressionCV(
                Cs=CLF_CV_CS, cv=3, penalty='l2', solver='lbfgs',
                max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)
            clf.fit(Xtr_s[:, sel], y_tr)
            pred = clf.predict(Xte_s[:, sel])

            acc_s[pi] += accuracy_score(y_te, pred)
            f1_s [pi] += f1_score(y_te, pred, zero_division=0)
            rec_s[pi] += recall_score(y_te, pred, zero_division=0)
            pre_s[pi] += precision_score(y_te, pred, zero_division=0)
            sigs [pi].append(sel.copy())

    return {
        'acc':  acc_s / N_FOLDS,
        'f1':   f1_s  / N_FOLDS,
        'rec':  rec_s / N_FOLDS,
        'pre':  pre_s / N_FOLDS,
        'sigs': sigs,
    }


# ─────────────────────────────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────
def evaluate(X, y, p, percentages):
    all_sigs = [[] for _ in percentages]
    shuffles = []
    for s in range(N_SHUFFLES):
        print(f"\n  == Shuffle {s+1:02d}/{N_SHUFFLES} ==", flush=True)
        t0 = time.time()
        sr = _one_shuffle(s, X, y, percentages, p)
        shuffles.append(sr)
        for pi in range(len(percentages)):
            all_sigs[pi].extend(sr['sigs'][pi])
        print(f"  Shuffle {s+1} done in {(time.time()-t0)/60:.1f} min")

    results = {k: [] for k in ['acc','f1','rec','pre','ki','ji','nogueira']}
    for pi in range(len(percentages)):
        for m in ['acc','f1','rec','pre']:
            results[m].append(float(np.mean([sr[m][pi] for sr in shuffles])))
        ki, ji   = compute_ki_ji(all_sigs[pi], p)
        nogueira = compute_nogueira(all_sigs[pi], p)
        results['ki']      .append(ki)
        results['ji']      .append(ji)
        results['nogueira'].append(nogueira)
    return results


# ─────────────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────
def print_results(results, percentages, p):
    bar = '=' * 108
    sep = '-' * 108
    print(f"\n{bar}")
    print(f"  STEP 1 / 10  |  InFusedLasso  (Bai et al., NeurIPS 2019)")
    print(f"  Formulation  :  min 1/2||y-Xb||^2 + l1||b||_1 + l2||Cb||_1 - l3 b'Ub  s.t. b>=0")
    print(f"  Solver       :  CCCP + CVXPY/CLARABEL (globally optimal convex sub-problems)")
    print(f"  Lambda sel.  :  {INNER_CV_FOLDS}-fold inner CV, data-driven grid (NO manual tuning)")
    print(f"  Shuffles     :  {N_SHUFFLES}  |  Outer folds: {N_FOLDS}")
    print(bar)
    print(f"  {'%Feat':>8}  {'k':>6}  {'Acc':>9}  {'F1':>9}  "
          f"{'Prec':>9}  {'Rec':>9}  {'KI':>9}  {'JI':>9}  {'Nogueira S':>11}")
    print(sep)
    for pi, pct in enumerate(percentages):
        k  = max(1, int(pct / 100.0 * p))
        ng = results['nogueira'][pi]
        ngs = f"{ng:>11.4f}" if not np.isnan(ng) else f"{'N/A':>11}"
        print(f"  {pct:>7.1f}%  {k:>6d}"
              f"  {results['acc'][pi]*100:>8.2f}%"
              f"  {results['f1'] [pi]*100:>8.2f}%"
              f"  {results['pre'][pi]*100:>8.2f}%"
              f"  {results['rec'][pi]*100:>8.2f}%"
              f"  {results['ki'][pi]:>9.4f}"
              f"  {results['ji'][pi]:>9.4f}"
              f"  {ngs}")
    print(sep)

    for tpct in [5.0, 10.0]:
        if tpct not in percentages: continue
        pi = percentages.index(tpct)
        k  = max(1, int(tpct / 100.0 * p))
        ng = results['nogueira'][pi]
        ngs = f"{ng:.4f}" if not np.isnan(ng) else "N/A"
        print(f"\n  --- Top {tpct:.0f}%  (k={k} features) ---")
        print(f"      Accuracy   : {results['acc'][pi]*100:.2f}%")
        print(f"      F1 Score   : {results['f1'] [pi]*100:.2f}%")
        print(f"      Precision  : {results['pre'][pi]*100:.2f}%")
        print(f"      Recall     : {results['rec'][pi]*100:.2f}%")
        print(f"      KI         : {results['ki'][pi]:.4f}")
        print(f"      JI         : {results['ji'][pi]:.4f}")
        print(f"      Nogueira S : {ngs}")
    print(f"\n{bar}")


def save_results(results, percentages, p):
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    payload = {
        'method':      'InFusedLasso',
        'n_shuffles':  N_SHUFFLES,
        'n_folds':     N_FOLDS,
        'percentages': percentages,
        'p':           p,
        'results': {
            k: [float(v) if not np.isnan(v) else None
                for v in lst]
            for k, lst in results.items()
        },
    }
    out = '/mnt/user-data/outputs/step1_infusedlasso_results.json'
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {out}")
    print("  In Step 2: import json; r=json.load(open('step1_infusedlasso_results.json'))")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    bar = '=' * 80
    print(bar)
    print("  STEP 1 / 10 -- InFusedLasso Feature Selection")
    print("  Formulation: min 1/2||y-Xb||^2+l1||b||_1+l2||Cb||_1-l3*b'Ub  s.t. b>=0")
    print(f"  Solver     : CCCP + CVXPY/CLARABEL")
    print(f"  CV config  : {INNER_CV_FOLDS}-fold inner CV, {N_LAM_3D}^3={N_LAM_3D**3} lambda-combos")
    print(f"  Shuffles   : {N_SHUFFLES}  (set N_SHUFFLES=20 for full results)")
    print(bar)

    X, y, p = load_data()

    print(f"\n[Running] {N_SHUFFLES} shuffles x {N_FOLDS} outer folds ...")
    results = evaluate(X, y, p, PERCENTAGES)

    print_results(results, PERCENTAGES, p)
    save_results(results, PERCENTAGES, p)

    elapsed = time.time() - t_start
    print(f"\n  Total wall time: {elapsed/60:.1f} min")
    print(bar)
    return results


if __name__ == '__main__':
    main()
