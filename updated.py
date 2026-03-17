"""
Implementation of: Fused lasso for feature selection using structural information
Optimization: Custom Split-Bregman Iteration (Paper Algorithm 1)
Dataset: 27 SCHZ / 27 CTRL Connectomes
"""

import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
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
    try:
        with h5py.File(FILE_PATH, 'r') as f:
            ctrl_ref = f['SC_FC_Connectomes/FC_correlation/ctrl']
            schz_ref = f['SC_FC_Connectomes/FC_correlation/schz']
            ctrl_mat = f[ctrl_ref[RESOLUTION_IDX, 0]][:]
            schz_mat = f[schz_ref[RESOLUTION_IDX, 0]][:]
    except FileNotFoundError:
        print("  [!] Dataset not found. Please verify the FILE_PATH.")
        # Returning dummy data for script validation if file is missing
        X = np.random.randn(54, 100)
        y = np.array([0]*27 + [1]*27, dtype=np.int32)
        return X, y

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
# INFUSEDLASSO - STRUCTURAL ALGORITHMS & SPLIT-BREGMAN
# ══════════════════════════════════════════════════════════════════════

def calculate_jsd(*distributions):
    """Jensen-Shannon Divergence for n distributions (Eq. 4 & 5)"""
    dists = np.array(distributions)
    mean_dist = np.mean(dists, axis=0)
    
    def entropy(P):
        return -np.sum(P * np.log(np.clip(P, 1e-12, 1.0)), axis=-1)
    
    # H_S(\sum \pi_i P_i) - \sum \pi_i H_S(P_i)
    jsd = entropy(mean_dist) - np.mean([entropy(d) for d in dists], axis=0)
    return np.clip(jsd, 0, None)  # Ensure non-negative due to float limits

def build_kernel_graph(f):
    """Kernel-based feature graph modeling (Eq. 3)"""
    # 1. Euclidean distance-based adjacency embedding A
    A = np.abs(f[:, None] - f[None, :])
    
    # 2. Kernel value computation using normalized dot product
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    A_safe = A / np.clip(A_norm, 1e-12, None)
    
    K = A_safe @ A_safe.T
    
    # 3. Convert to Probability Distribution format
    K_flat = K.flatten()
    return K_flat / (np.sum(K_flat) + 1e-12)

def compute_U_and_order(X, y):
    n, p = X.shape
    G = np.zeros((p, n * n), dtype=np.float32)
    G_hat = np.zeros((p, n * n), dtype=np.float32)
    classes = np.unique(y)

    # Calculate continuous value based target feature
    f_hat_matrix = np.zeros((n, p), dtype=np.float32)
    for c in classes:
        mask = (y == c)
        f_hat_matrix[mask, :] = np.mean(X[mask, :], axis=0)

    # Construct individual graphs G_i and G_hat_i
    for i in range(p):
        G[i] = build_kernel_graph(X[:, i])
        G_hat[i] = build_kernel_graph(f_hat_matrix[:, i])

    # Calculate Individual Relevance (Eq. 5 for n=2)
    relevance = np.zeros(p)
    for i in range(p):
        jsd_i = calculate_jsd(G[i], G_hat[i])
        relevance[i] = np.exp(-jsd_i)

    # Reorder features descending by relevance
    order = np.argsort(relevance)[::-1]
    
    G_ordered = G[order]
    G_hat_ordered = G_hat[order]

    # Compute Structural Interaction Measure U (Eq. 6)
    U = np.zeros((p, p), dtype=np.float32)
    for i in range(p):
        for j in range(i, p):
            I_S_G_i_j = np.exp(-calculate_jsd(G_ordered[i], G_ordered[j]))
            I_S_G_i_j_hat_i = np.exp(-calculate_jsd(G_ordered[i], G_ordered[j], G_hat_ordered[i]))
            I_S_G_i_j_hat_j = np.exp(-calculate_jsd(G_ordered[i], G_ordered[j], G_hat_ordered[j]))
            
            U_val = (I_S_G_i_j_hat_i + I_S_G_i_j_hat_j) / np.clip(I_S_G_i_j, 1e-12, None)
            U[i, j] = U[j, i] = U_val

    return U, order

def soft_threshold(x, lam):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def split_bregman_fused_lasso(X, y, U, lambda1=0.1, lambda2=0.1, lambda3=0.01, max_iter=200, tol=1e-4):
    """Algorithm 1: Iterative Split Bregman Optimization"""
    n, p = X.shape
    
    mu1, mu2 = 1.0, 1.0 
    delta1, delta2 = mu1, mu2 
    
    # Fused Lasso difference matrix C (Eq. 8 / Section 4.2)
    C = np.eye(p - 1, p, k=0) - np.eye(p - 1, p, k=1)
    
    # Construct D = X^T X - 2*lambda3*U + mu1*I + mu2*C^T C
    XtX = X.T @ X
    CtC = C.T @ C
    D = XtX - 2 * lambda3 * U + mu1 * np.eye(p) + mu2 * CtC
    
    # Stability Check: Enforce strictly positive-definite D for invertibility
    eigvals = np.linalg.eigvalsh(D)
    min_eig = np.min(eigvals)
    if min_eig < 1e-5:
        # Inject ridge penalty to ensure matrix is well-conditioned
        D += (1e-5 - min_eig) * np.eye(p)
        
    D_inv = np.linalg.inv(D)
    X_ty = X.T @ y
    
    # Initialize optimization variables
    beta = np.zeros(p)
    p_var = np.zeros(p)
    q_var = np.zeros(p - 1)
    u_var = np.zeros(p)
    v_var = np.zeros(p - 1)
    
    for k in range(max_iter):
        # 1. Update beta (Eq. 16)
        rhs = X_ty + mu1 * (p_var - u_var / mu1) + mu2 * C.T @ (q_var - v_var / mu2)
        beta_next = D_inv @ rhs
        
        # 2. Update p (Eq. 17)
        p_next = soft_threshold(beta_next + u_var / mu1, lambda1 / mu1)
        
        # 3. Update q (Eq. 18)
        q_next = soft_threshold(C @ beta_next + v_var / mu2, lambda2 / mu2)
        
        # 4. Update u (Eq. 19)
        u_next = u_var + delta1 * (beta_next - p_next)
        
        # 5. Update v (Eq. 20)
        v_next = v_var + delta2 * (C @ beta_next - q_next)
        
        # Convergence Check
        if np.linalg.norm(beta_next - beta) < tol:
            beta = beta_next
            break
            
        beta, p_var, q_var = beta_next, p_next, q_next
        u_var, v_var = u_next, v_next

    return beta

def rank_infused_lasso(Xs, y):
    U, order = compute_U_and_order(Xs, y)
    
    # Use reordered features
    Xs_ordered = Xs[:, order]
    y_num = np.where(y == 0, -1.0, 1.0) 
    
    # Apply Split-Bregman directly on reordered data
    beta_ordered = split_bregman_fused_lasso(Xs_ordered, y_num, U)
    
    # Map back to original indices to sort by absolute beta weight
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
            Xtr_s = StandardScaler().fit_transform(X[tr_idx])
            Xte_s = StandardScaler().fit_transform(X[te_idx])
            
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
    print(f'\nRunning {N_SHUFFLES}x{N_FOLDS} CV with Split-Bregman Optimization ...')
    results = evaluate(X, y, PERCENTAGES)

    print("\n  InFusedLasso (Split-Bregman): Performance & Stability")
    print(f"  {'%Features':>10}  {'KI':>9}  {'JI':>9}  {'Nogueira':>10}  {'Accuracy':>10}  {'F1-Score':>10}")
    print("─"*70)
    for pi, pct in enumerate(PERCENTAGES):
        print(f"  {pct:>10.1f}%  {results['ki'][pi]:>9.4f}  {results['ji'][pi]:>9.4f}  {results['nog'][pi]:>10.4f}  {results['acc'][pi]*100:>9.2f}%  {results['f1'][pi]*100:>9.2f}%")

if __name__ == '__main__':
    main()
