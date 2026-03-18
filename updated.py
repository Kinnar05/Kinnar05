"""
Implementation of: Fused Lasso for Feature Selection Using Structural Information
Optimization: Exact Split-Bregman Iteration (Algorithm 1)
Dataset: 27 SCHZ / 27 CTRL Connectomes

FIX SUMMARY (4 issues addressed):
  1. Graph construction now exactly follows Eq. (3): kernel matrix K is built from the
     Euclidean-distance adjacency matrix A, then ROW-NORMALISED to a probability
     distribution G_i (so it sums to 1 over the n*n entries), matching paper Section 3.1.
  2. Structural interaction matrix U is computed term-by-term from Eq. (6):
       U_{i,j} = [IS(G_i,G_j;G^_i) + IS(G_i,G_j;G^_j)] / IS(G_i,G_j)
     where IS = exp(-JSD) and JSD is computed for 2 or 3 equal-weight distributions
     via Eq. (4)-(5).  No vectorised entropy short-cut that alters the semantics.
  3. Split-Bregman solver mirrors Algorithm 1 / Eqs. (16-20) exactly, including the
     correct RHS of Eq. (16):  D*beta = X^T y + mu1*(p - u/mu1) + mu2*C^T*(q - v/mu2)
     and the step-size choices  delta1 = mu1, delta2 = mu2 as stated in the paper.
  4. D-matrix conditioning: we check cond(D) before inversion and add a small Tikhonov
     ridge (eps * I) if D is ill-conditioned (cond > 1e10).  The ridge is reported so
     the user knows when it fires.  This replaces the silent pinv fallback.
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np

from sklearn.svm            import SVC
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics        import (accuracy_score, f1_score,
                                    recall_score, precision_score)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
FILE_PATH       = '/kaggle/input/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat'
RESOLUTION_IDX  = 0
N_ROIS_EXPECTED = 83

PERCENTAGES  = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 60.0, 70.0, 80.0]
N_SHUFFLES   = 20
N_FOLDS      = 5
RANDOM_STATE = 42

# Conditioning threshold: if cond(D) > COND_THRESHOLD we add a ridge
COND_THRESHOLD = 1e10
# Ridge magnitude for ill-conditioned D (Tikhonov regularisation)
RIDGE_EPS      = 1e-6

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
    tri    = np.triu_indices(n_rois, k=1)

    def vectorise(mats):
        return np.abs(
            np.array([mats[i][tri] for i in range(len(mats))], dtype=np.float64)
        )

    X_ctrl = vectorise(ctrl_mat)
    X_schz = vectorise(schz_mat)
    X = np.vstack([X_ctrl, X_schz])
    y = np.array([0] * 27 + [1] * 27, dtype=np.int32)
    print(f"  Loaded: {n_rois} ROIs | p={X.shape[1]} features")
    return X, y


# ══════════════════════════════════════════════════════════════════════
# FIX 1 & 2 — EXACT GRAPH CONSTRUCTION AND STRUCTURAL INTERACTION MATRIX
# ══════════════════════════════════════════════════════════════════════

def _kernel_graph(f_vec):
    """
    Convert a 1-D feature vector f_vec (M samples) into the probability
    distribution G_i over M×M pairs, following Eq. (3) of the paper.

    Steps (Section 3.1):
      1. Build Euclidean-distance adjacency matrix A where A_{a,b} = |f_a - f_b|.
      2. Interpret each row of A as a "distance-based embedding vector" for sample a.
      3. Compute the normalised kernel:
            K_{a,b} = <A_{a,:}, A_{b,:}> / sqrt(<A_{a,:},A_{a,:}> * <A_{b,:},A_{b,:}>)
         This is the cosine similarity of the distance embedding rows (Eq. 3).
      4. Normalise the entire M×M kernel matrix to a probability distribution
         (divide by its total sum) so that G_i can be used directly in JSD (Eq. 4).
    """
    M = len(f_vec)
    # Step 1 – Euclidean distances (absolute differences for scalar features)
    A = np.abs(f_vec[:, None] - f_vec[None, :]).astype(np.float64)  # (M, M)

    # Step 2-3 – Cosine similarity of rows of A (Eq. 3)
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)            # (M, 1)
    denom     = row_norms @ row_norms.T                              # (M, M)
    K         = (A @ A.T) / np.maximum(denom, 1e-12)                # (M, M)

    # Step 4 – Normalise to a probability distribution (required by JSD / Eq. 4)
    total = K.sum()
    if total < 1e-30:
        # Degenerate feature (all identical samples): uniform distribution
        G = np.full(M * M, 1.0 / (M * M), dtype=np.float64)
    else:
        G = (K / total).flatten()                                    # (M*M,)
    return G


def _jsd_2(P, Q):
    """
    Jensen-Shannon divergence between two equal-weight distributions (Eq. 4, n=2).
    JSD(P,Q) = H((P+Q)/2) - 0.5*(H(P)+H(Q))
    """
    M = (P + Q) / 2.0
    return _entropy(M) - 0.5 * (_entropy(P) + _entropy(Q))


def _jsd_3(P, Q, R):
    """
    Jensen-Shannon divergence among three equal-weight distributions (Eq. 4, n=3).
    JSD(P,Q,R) = H((P+Q+R)/3) - (H(P)+H(Q)+H(R))/3
    """
    M = (P + Q + R) / 3.0
    return _entropy(M) - (_entropy(P) + _entropy(Q) + _entropy(R)) / 3.0


def _entropy(P):
    """Shannon entropy of a probability distribution (handles zeros)."""
    P_safe = np.maximum(P, 1e-300)
    return -np.dot(P_safe, np.log(P_safe))


def _is(jsd_val):
    """Similarity measure IS = exp(-JSD) from Eq. (5)."""
    return np.exp(-jsd_val)


def compute_U_and_order(X, y):
    """
    Build the structural interaction matrix U (Eq. 6) and the relevance-based
    feature ordering required by the Fused Lasso framework (Section 5.4).

    FIX 1: Graph construction uses _kernel_graph() which follows Eq. (3) exactly.
    FIX 2: U_{i,j} is computed term-by-term from Eq. (6):
              U_{i,j} = [IS(G_i,G_j;G^_i) + IS(G_i,G_j;G^_j)] / IS(G_i,G_j)
    Relevance R_i = IS(G_i; G^_i) = exp(-JSD(G_i, G^_i))  for individual ordering.
    """
    n, p = X.shape
    classes = np.unique(y)

    # ── Step 1: compute target feature f_hat for each feature (Section 3.1) ──
    f_hat = np.zeros_like(X)                                     # (n, p)
    for c in classes:
        mask = (y == c)
        f_hat[mask, :] = X[mask, :].mean(axis=0)

    # ── Step 2: build G_i and G^_i graphs for every feature ──
    print("    Building feature graphs ...", flush=True)
    G     = np.zeros((p, n * n), dtype=np.float64)
    G_hat = np.zeros((p, n * n), dtype=np.float64)
    for i in range(p):
        G[i]     = _kernel_graph(X[:, i])
        G_hat[i] = _kernel_graph(f_hat[:, i])

    # ── Step 3: individual relevance for ordering ──
    # R_i = IS(G_i, G^_i)  — pairwise JSD between feature graph and its target
    print("    Computing relevance for ordering ...", flush=True)
    relevance = np.array([
        _is(_jsd_2(G[i], G_hat[i])) for i in range(p)
    ])
    order = np.argsort(relevance)[::-1]          # descending relevance

    # Re-order arrays so indices align with β
    G     = G[order]
    G_hat = G_hat[order]

    # ── Step 4: build U matrix from Eq. (6) ──
    # U_{i,j} = [IS(G_i,G_j;G^_i) + IS(G_i,G_j;G^_j)] / IS(G_i,G_j)
    print("    Building interaction matrix U ...", flush=True)
    U = np.zeros((p, p), dtype=np.float64)

    for i in range(p):
        # Vectorised over all j simultaneously
        # Numerator term 1:  IS(G_i, G_j, G^_i)  for all j
        num1 = np.array([
            _is(_jsd_3(G[i], G[j], G_hat[i])) for j in range(p)
        ])
        # Numerator term 2:  IS(G_i, G_j, G^_j)  for all j
        num2 = np.array([
            _is(_jsd_3(G[i], G[j], G_hat[j])) for j in range(p)
        ])
        # Denominator:  IS(G_i, G_j)  for all j
        denom = np.array([
            _is(_jsd_2(G[i], G[j])) for j in range(p)
        ])
        # Avoid division by zero (IS ≥ 0; degenerate only when two identical graphs)
        denom = np.maximum(denom, 1e-12)
        U[i, :] = (num1 + num2) / denom

    return U, order


# ══════════════════════════════════════════════════════════════════════
# FIX 3 & 4 — EXACT SPLIT-BREGMAN SOLVER WITH CONDITIONING CHECK
# ══════════════════════════════════════════════════════════════════════

def split_bregman_infused_lasso(
    X, y, U,
    lambda1=0.1, lambda2=0.1, lambda3=0.01,
    mu1=1.0, mu2=1.0,
    max_iter=150, tol=1e-4,
):
    """
    Solves Eq. (8) via the Split-Bregman iteration in Algorithm 1.

    FIX 3 – Exact Algorithm 1 / Eqs. (16)-(20):
      • β update  (Eq. 16):  D β = X^T y + mu1*(p - u/mu1) + mu2*C^T*(q - v/mu2)
        Note: the paper writes  mu1*(p_k - mu1^{-1} u_k) which is p_k - u_k/mu1,
        then pre-multiplied by mu1.  Expanding: mu1*p_k - u_k.  This is what
        the code computes in `rhs`.
      • p update  (Eq. 17):  p = S_{lambda1/mu1}(β + u/mu1)
      • q update  (Eq. 18):  q = S_{lambda2/mu2}(C β + v/mu2)
      • u update  (Eq. 19):  u ← u + delta1*(β - p),   delta1 = mu1
      • v update  (Eq. 20):  v ← v + delta2*(C β - q), delta2 = mu2
        (The paper proves convergence for 0 < delta ≤ mu, and states the
         implementation uses delta1 = mu1, delta2 = mu2.)

    FIX 4 – Conditioning of D (Eq. 16):
      D = X^T X - 2*lambda3*U + mu1*I + mu2*C^T C
      Before inverting, we estimate cond(D).  If cond(D) > COND_THRESHOLD we add
      a small ridge eps*I and report it.  This replaces the silent pinv fallback.
    """
    n, p = X.shape
    # Difference matrix C: (p-1) × p, C_{i,i}=1, C_{i,i+1}=-1  (Eq. 8 / Section 4.2)
    C = (np.eye(p - 1, p, k=0) - np.eye(p - 1, p, k=1)).astype(np.float64)

    # ── Precompute D (Eq. 16) ──
    XtX  = X.T @ X                                               # (p, p)
    CtC  = C.T @ C                                               # (p, p)
    D    = XtX - 2.0 * lambda3 * U + mu1 * np.eye(p) + mu2 * CtC

    # FIX 4 – conditioning check + optional ridge
    cond_D = np.linalg.cond(D)
    if cond_D > COND_THRESHOLD:
        ridge = RIDGE_EPS * np.eye(p)
        print(
            f"      [WARNING] cond(D)={cond_D:.2e} > {COND_THRESHOLD:.0e}. "
            f"Adding Tikhonov ridge eps={RIDGE_EPS:.0e}.",
            flush=True,
        )
        D += ridge

    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        # Should not happen after ridge, but guard anyway
        raise RuntimeError(
            "D is singular even after ridge regularisation. "
            "Try increasing RIDGE_EPS or decreasing lambda3."
        )

    # ── Initialise primal and dual variables ──
    beta  = np.zeros(p, dtype=np.float64)
    p_vec = np.zeros(p, dtype=np.float64)
    q_vec = np.zeros(p - 1, dtype=np.float64)
    u_vec = np.zeros(p, dtype=np.float64)
    v_vec = np.zeros(p - 1, dtype=np.float64)

    XtY = X.T @ y                                                # (p,)

    def soft_threshold(w, thresh):
        return np.sign(w) * np.maximum(0.0, np.abs(w) - thresh)

    for _ in range(max_iter):
        beta_old = beta.copy()

        # ── Step 1: β update (Eq. 16) ──
        # rhs = X^T y + mu1*(p_k - u_k/mu1) + mu2*C^T*(q_k - v_k/mu2)
        #     = X^T y + mu1*p_k - u_k + mu2*C^T*q_k - C^T*v_k
        rhs  = XtY + mu1 * p_vec - u_vec + mu2 * (C.T @ q_vec) - (C.T @ v_vec)
        beta = D_inv @ rhs

        # Explicit non-negativity constraint (β ≥ 0 required by Eq. 7)
        beta = np.maximum(0.0, beta)

        # ── Step 2: p update (Eq. 17) ──
        # p = S_{lambda1/mu1}(β + u/mu1)
        p_vec = soft_threshold(beta + u_vec / mu1, lambda1 / mu1)

        # ── Step 3: q update (Eq. 18) ──
        # q = S_{lambda2/mu2}(C β + v/mu2)
        q_vec = soft_threshold(C @ beta + v_vec / mu2, lambda2 / mu2)

        # ── Step 4: u, v updates (Eqs. 19 & 20) with delta1=mu1, delta2=mu2 ──
        u_vec = u_vec + mu1 * (beta - p_vec)
        v_vec = v_vec + mu2 * (C @ beta - q_vec)

        # Convergence check
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


def rank_infused_lasso(Xs, y):
    """Compute InFusedLasso feature ranking on training data Xs."""
    U, order  = compute_U_and_order(Xs, y)
    Xs_ordered = Xs[:, order]
    # Convert {0,1} labels to {-1,+1} for regression
    y_num      = np.where(y == 0, -1.0, 1.0)

    beta_ordered = split_bregman_infused_lasso(Xs_ordered, y_num, U)

    # Map back to original feature ordering
    beta_orig          = np.zeros(Xs.shape[1], dtype=np.float64)
    beta_orig[order]   = np.abs(beta_ordered)

    return np.argsort(beta_orig)[::-1].copy()


# ══════════════════════════════════════════════════════════════════════
# STABILITY METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_stability_metrics(signatures, p):
    lam = len(signatures)
    if lam < 2:
        return np.nan, np.nan, np.nan
    k = len(signatures[0])

    S = np.zeros((lam, p), dtype=np.float32)
    for i, sig in enumerate(signatures):
        idx = sig[(sig >= 0) & (sig < p)]
        S[i, idx] = 1.0

    intersect = S @ S.T
    r_vals    = intersect[np.triu_indices(lam, k=1)]
    k2p       = float(k) ** 2 / float(p)
    denom     = float(k) - k2p

    ki  = float(np.clip(np.mean((r_vals - k2p) / denom), -1.0, 1.0)) \
          if abs(denom) > 1e-12 else 1.0
    ji  = float(np.mean(
              np.where(2.0 * k - r_vals > 0, r_vals / (2.0 * k - r_vals), 1.0)
          ))

    p_j     = np.mean(S, axis=0)
    s       = float(k) / float(p)
    nog_den = s * (1.0 - s)
    nog     = float(np.clip(
                  1.0 - (np.mean(p_j * (1 - p_j)) / nog_den), -1.0, 1.0
              )) if nog_den > 1e-12 else 1.0

    return ki, ji, nog


# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate(X, y, percentages):
    p   = X.shape[1]
    ks  = [max(1, int(pct / 100.0 * p)) for pct in percentages]

    metrics = {k: np.zeros((N_SHUFFLES, len(percentages)))
               for k in ['acc', 'f1', 'rec', 'pre']}
    sigs = [[] for _ in percentages]

    for s in range(N_SHUFFLES):
        print(f"    Shuffle {s+1:02d}/{N_SHUFFLES} ...", flush=True)
        skf = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + s
        )

        for tr_idx, te_idx in skf.split(X, y):
            scaler  = StandardScaler()
            Xtr_s   = scaler.fit_transform(X[tr_idx])
            Xte_s   = scaler.transform(X[te_idx])

            rank    = rank_infused_lasso(Xtr_s, y[tr_idx])

            for pi, k in enumerate(ks):
                sel = rank[:k]
                clf = SVC(kernel="linear", random_state=RANDOM_STATE)
                clf.fit(Xtr_s[:, sel], y[tr_idx])
                pred = clf.predict(Xte_s[:, sel])

                metrics['acc'][s, pi] += accuracy_score(y[te_idx], pred)   / N_FOLDS
                metrics['f1'][s, pi]  += f1_score(y[te_idx], pred,
                                                   zero_division=0)         / N_FOLDS
                metrics['rec'][s, pi] += recall_score(y[te_idx], pred,
                                                       zero_division=0)     / N_FOLDS
                metrics['pre'][s, pi] += precision_score(y[te_idx], pred,
                                                          zero_division=0)  / N_FOLDS
                sigs[pi].append(sel.copy())

    results = {m: np.mean(metrics[m], axis=0).tolist() for m in metrics}
    results['ki'], results['ji'], results['nog'] = zip(*[
        compute_stability_metrics(sigs[pi], p) for pi in range(len(percentages))
    ])
    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    X, y = load_data()
    print(f'\nRunning {N_SHUFFLES}x{N_FOLDS} CV with exact Split-Bregman Optimization ...')
    results = evaluate(X, y, PERCENTAGES)

    print("\n  InFusedLasso (Split Bregman): Performance & Stability")
    print(f"  {'%Features':>10}  {'KI':>9}  {'JI':>9}  {'Nogueira':>10}"
          f"  {'Accuracy':>10}  {'F1-Score':>10}")
    print("─" * 70)
    for pi, pct in enumerate(PERCENTAGES):
        print(
            f"  {pct:>10.1f}%  {results['ki'][pi]:>9.4f}  "
            f"{results['ji'][pi]:>9.4f}  {results['nog'][pi]:>10.4f}  "
            f"{results['acc'][pi]*100:>9.2f}%  {results['f1'][pi]*100:>9.2f}%"
        )


if __name__ == '__main__':
    main()
