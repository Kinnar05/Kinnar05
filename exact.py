"""
Implementation of: Fused lasso for feature selection using structural information
Optimization: Exact Split-Bregman Iteration (Algorithm 1)
Dataset: 27 SCHZ / 27 CTRL Connectomes
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib

from sklearn.svm               import SVC
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
# INFUSEDLASSO - STRUCTURAL ALGORITHMS & SPLIT-BREGMAN OPTIMIZATION
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

def split_bregman_infused_lasso(X, y, U, lambda1=0.1, lambda2=0.1, lambda3=0.01, max_iter=100, tol=1e-4):
    """
    Solves Eq. 8 exactly using the Split Bregman Iteration detailed in Algorithm 1.
    No convex projection mapping is applied here; solves the true non-convex form.
    """
    n, p = X.shape
    C = np.eye(p - 1, p, k=0) - np.eye(p - 1, p, k=1)
    
    # Setup Split Bregman Hyperparameters
    mu1, mu2 = 1.0, 1.0
    
    # 1. Precompute D matrix and its inverse (Eq. 16)
    # D = X^T X - 2*lambda3*U + mu1*I + mu2*C^T C
    D = X.T @ X - 2 * lambda3 * U + mu1 * np.eye(p) + mu2 * (C.T @ C)
    
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        D_inv = np.linalg.pinv(D)
        
    # Initialize Primal and Dual variables
    beta = np.zeros(p)
    p_vec = np.zeros(p)
    q_vec = np.zeros(p - 1)
    u_vec = np.zeros(p)
    v_vec = np.zeros(p - 1)
    
    X_Ty = X.T @ y
    
    def soft_threshold(w, thresh):
        return np.sign(w) * np.maximum(0, np.abs(w) - thresh)

    for _ in range(max_iter):
        beta_old = beta.copy()
        
        # Step 1: Update beta (Eq. 16)
        right_side = X_Ty + mu1 * (p_vec - u_vec / mu1) + mu2 * C.T @ (q_vec - v_vec / mu2)
        beta = D_inv @ right_side
        
        # Explicit Non-negativity constraint
        beta = np.maximum(0, beta)
        
        # Step 2: Update p (Eq. 17)
        p_vec = soft_threshold(beta + u_vec / mu1, lambda1 / mu1)
        
        # Step 3: Update q (Eq. 18)
        q_vec = soft_threshold(C @ beta + v_vec / mu2, lambda2 / mu2)
        
        # Step 4: Update u, v (Eq. 19 & 20)
        # Using delta1 = mu1, delta2 = mu2 as proven to converge in the paper
        u_vec = u_vec + mu1 * (beta - p_vec)
        v_vec = v_vec + mu2 * (C @ beta - q_vec)
        
        # Convergence Check
        if np.linalg.norm(beta - beta_old) < tol:
            break
            
    return beta

def rank_infused_lasso(Xs, y):
    U, order = compute_U_and_order(Xs, y)
    Xs_ordered = Xs[:, order]
    y_num = np.where(y == 0, -1.0, 1.0) 
    
    beta_ordered = split_bregman_infused_lasso(Xs_ordered, y_num, U)
    
    beta_orig = np.zeros(Xs.shape[1], dtype=np.float64)
    beta_orig[order] = np.abs(beta_ordered)
    
    return np.argsort(beta_orig)[::-1].copy()

# ══════════════════════════════════════════════════════════════════════
# STABILITY METRICS & EVALUATION
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
            # FIXED: Fit the scaler strictly on the train set, transform test
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(X[tr_idx])
            Xte_s = scaler.transform(X[te_idx])
            
            rank = rank_infused_lasso(Xtr_s, y[tr_idx])
            
            for pi, k in enumerate(ks):
                sel = rank[:k]
                # FIXED: Swapped Logistic Regression for Linear SVM per Section 5.1
                clf = SVC(kernel="linear", random_state=RANDOM_STATE)
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
    print(f'\nRunning {N_SHUFFLES}x{N_FOLDS} CV with exact Split-Bregman Optimization ...')
    results = evaluate(X, y, PERCENTAGES)

    print("\n  InFusedLasso (Split Bregman): Performance & Stability")
    print(f"  {'%Features':>10}  {'KI':>9}  {'JI':>9}  {'Nogueira':>10}  {'Accuracy':>10}  {'F1-Score':>10}")
    print("─"*70)
    for pi, pct in enumerate(PERCENTAGES):
        print(f"  {pct:>10.1f}%  {results['ki'][pi]:>9.4f}  {results['ji'][pi]:>9.4f}  {results['nog'][pi]:>10.4f}  {results['acc'][pi]*100:>9.2f}%  {results['f1'][pi]*100:>9.2f}%")

if __name__ == '__main__':
    main()
