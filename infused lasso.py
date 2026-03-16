"""
Implementation of: Fused lasso for feature selection using structural information
Optimization: Solved completely using CVXPY (Projected to PSD)
Dataset: 27 SCHZ / 27 CTRL Connectomes
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.subplots as plt

from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import StratifiedKFold
from sklearn.metrics           import accuracy_score, f1_score, recall_score, precision_score

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
FILE_PATH      = '/kaggle/input/schrinzophenia/27_SCHZ_CTRL_dataset(1).mat'
RESOLUTION_IDX = 0          
N_ROIS_EXPECTED= 83

PERCENTAGES = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 60.0, 70.0, 80.0]

N_SHUFFLES   = 20
N_FOLDS      = 5
RANDOM_STATE = 42

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
    tri = np.triu_indices(n_rois, k=1)

    def vectorise(mats):
        return np.abs(np.array([mats[i][tri] for i in range(len(mats))], dtype=np.float64))

    X_ctrl = vectorise(ctrl_mat)
    X_schz = vectorise(schz_mat)
    X = np.vstack([X_ctrl, X_schz])
    y = np.array([0]*27 + [1]*27, dtype=np.int32)

    print(f"  Loaded: {n_rois} ROIs | p={X.shape[1]} features")
    return X, y

# ══════════════════════════════════════════════════════════════════════
# INFUSEDLASSO - STRUCTURAL ALGORITHMS & CVXPY OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════

def compute_U_and_order(X, y):
    n, p = X.shape
    G = np.zeros((p, n * n), dtype=np.float32)
    G_hat = np.zeros((p, n * n), dtype=np.float32)
    classes = np.unique(y)

    f_hat_matrix = np.zeros((n, p), dtype=np.float32)
    for c in classes:
        mask = (y == c)
        f_hat_matrix[mask, :] = np.mean(X[mask, :], axis=0)

    for i in range(p):
        fi = X[:, i]
        A = np.abs(fi[:, None] - fi[None, :])
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
        K = A_norm @ A_norm.T
        G[i] = (K / (np.sum(K) + 1e-10)).flatten()

        f_hat = f_hat_matrix[:, i]
        A_hat = np.abs(f_hat[:, None] - f_hat[None, :])
        A_hat_norm = A_hat / (np.linalg.norm(A_hat, axis=1, keepdims=True) + 1e-10)
        K_hat = A_hat_norm @ A_hat_norm.T
        G_hat[i] = (K_hat / (np.sum(K_hat) + 1e-10)).flatten()

    def entropy(P):
        return -np.sum(P * np.log(np.clip(P, 1e-12, 1.0)), axis=-1)

    H_G = entropy(G)
    H_G_hat = entropy(G_hat)

    M_i_hat = (G + G_hat) / 2.0
    Relevance = np.exp(-(entropy(M_i_hat) - 0.5 * (H_G + H_G_hat)))
    
    order = np.argsort(Relevance)[::-1]
    
    G, G_hat = G[order], G_hat[order]
    H_G, H_G_hat = H_G[order], H_G_hat[order]

    U = np.zeros((p, p), dtype=np.float32)
    for i in range(p):
        I_S_2 = np.exp(-(entropy((G[i] + G) / 2.0) - 0.5 * (H_G[i] + H_G)))
        I_S_3_i = np.exp(-(entropy((G[i] + G + G_hat[i]) / 3.0) - (H_G[i] + H_G + H_G_hat[i]) / 3.0))
        I_S_3_j = np.exp(-(entropy((G[i] + G + G_hat) / 3.0) - (H_G[i] + H_G + H_G_hat) / 3.0))
        U[i, :] = (I_S_3_i + I_S_3_j) / np.clip(I_S_2, 1e-12, 1.0)

    return U, order

def cvxpy_infused_lasso(X, y, U, lambda1=0.1, lambda2=0.1, lambda3=0.01):
    """Solves Eq. 8 completely using CVXPY by projecting the Hessian to PSD"""
    n, p = X.shape
    C = np.eye(p - 1, p, k=0) - np.eye(p - 1, p, k=1)

    # 1. Isolate the quadratic terms to form the Hessian Matrix H
    # Eq 8 Expansion: 0.5 * beta^T (X^T X - 2*lambda3*U) beta
    H = X.T @ X - 2 * lambda3 * U

    # 2. Project H to Positive Semi-Definite space for CVXPY (DCP rules)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, 1e-6) # Clip negative eigenvalues
    H_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # 3. Formulate Convex Problem
    beta = cp.Variable(p)
    q = X.T @ y

    # Objective: 0.5 * beta^T H_psd beta - q^T beta + L1 Penalties
    objective = cp.Minimize(
        0.5 * cp.quad_form(beta, H_psd) - q.T @ beta +
        lambda1 * cp.norm1(beta) + 
        lambda2 * cp.norm1(C @ beta)
    )
    
    prob = cp.Problem(objective)
    # OSQP is optimized for sparse convex programs and L1 penalties
    prob.solve(solver=cp.OSQP, max_iter=2500)
    
    if beta.value is None:
        return np.zeros(p) # Fallback if solver fails to converge
    return beta.value

def rank_infused_lasso(Xs, y):
    U, order = compute_U_and_order(Xs, y)
    Xs_ordered = Xs[:, order]
    y_num = np.where(y == 0, -1.0, 1.0) 
    
    beta_ordered = cvxpy_infused_lasso(Xs_ordered, y_num, U)
    
    beta_orig = np.zeros(Xs.shape[1], dtype=np.float64)
    beta_orig[order] = np.abs(beta_ordered)
    
    return np.argsort(beta_orig)[::-1].copy()

# ══════════════════════════════════════════════════════════════════════
# STABILITY METRICS & EVALUATION (Abridged for layout)
# ══════════════════════════════════════════════════════════════════════

def compute_stability_metrics(signatures, p):
    lam = len(signatures)
    if lam < 2: return np.nan, np.nan, np.nan
    k = len(signatures[0])
    
    S = np.zeros((lam, p), dtype=np.float32)
    for i, sig in enumerate(signatures):
        S[i, sig[(sig >= 0) & (sig < p)]] = 1.0

    intersect = S @ S.T
    r_vals = intersect[np.triu_indices(lam, k=1)]
    k2p = float(k)**2 / float(p)
    denom = float(k) - k2p
    
    ki = float(np.clip(np.mean((r_vals - k2p) / denom), -1.0, 1.0)) if abs(denom) > 1e-12 else 1.0
    ji = float(np.mean(np.where(2.0 * k - r_vals > 0, r_vals / (2.0 * k - r_vals), 1.0)))

    p_j = np.mean(S, axis=0)
    s = float(k) / float(p)
    nog_denom = s * (1.0 - s)
    nog = float(np.clip(1.0 - (np.mean(p_j * (1 - p_j)) / nog_denom), -1.0, 1.0)) if nog_denom > 1e-12 else 1.0

    return ki, ji, nog

def evaluate(X, y, percentages):
    p = X.shape[1]
    ks = [max(1, int(pct / 100.0 * p)) for pct in percentages]
    
    metrics = {k: np.zeros((N_SHUFFLES, len(percentages))) for k in ['acc', 'f1', 'rec', 'pre']}
    sigs = [[] for _ in percentages]
    
    for s in range(N_SHUFFLES):
        print(f"    Shuffle {s+1:02d}/{N_SHUFFLES} ...", flush=True)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + s)
        
        for tr_idx, te_idx in skf.split(X, y):
            Xtr_s = StandardScaler().fit_transform(X[tr_idx])
            Xte_s = StandardScaler().fit_transform(X[te_idx]) # Re-fit to match scope
            
            rank = rank_infused_lasso(Xtr_s, y[tr_idx])
            
            for pi, k in enumerate(ks):
                sel = rank[:k]
                clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE)
                clf.fit(Xtr_s[:, sel], y[tr_idx])
                pred = clf.predict(Xte_s[:, sel])

                metrics['acc'][s, pi] += accuracy_score(y[te_idx], pred) / N_FOLDS
                metrics['f1'][s, pi]  += f1_score(y[te_idx], pred, zero_division=0) / N_FOLDS
                metrics['rec'][s, pi] += recall_score(y[te_idx], pred, zero_division=0) / N_FOLDS
                metrics['pre'][s, pi] += precision_score(y[te_idx], pred, zero_division=0) / N_FOLDS
                sigs[pi].append(sel.copy())

    results = {m: np.mean(metrics[m], axis=0).tolist() for m in metrics}
    results['ki'], results['ji'], results['nog'] = zip(*[compute_stability_metrics(sigs[pi], p) for pi in range(len(percentages))])
    
    return results

# ══════════════════════════════════════════════════════════════════════
# MAIN ROUTINE
# ══════════════════════════════════════════════════════════════════════
def main():
    X, y = load_data()
    print(f'\nRunning {N_SHUFFLES}x{N_FOLDS} CV with CVXPY (Projected PSD Hessian) ...')
    results = evaluate(X, y, PERCENTAGES)

    print("\n  InFusedLasso (CVXPY): Performance & Stability")
    print(f"  {'%Features':>10}  {'KI':>9}  {'JI':>9}  {'Nogueira':>10}  {'Accuracy':>10}  {'F1-Score':>10}")
    print("─"*70)
    for pi, pct in enumerate(PERCENTAGES):
        print(f"  {pct:>10.1f}%  {results['ki'][pi]:>9.4f}  {results['ji'][pi]:>9.4f}  {results['nog'][pi]:>10.4f}  {results['acc'][pi]*100:>9.2f}%  {results['f1'][pi]*100:>9.2f}%")

if __name__ == '__main__':
    main()
