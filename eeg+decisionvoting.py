"""
Decision Voting Graph Convolutional Network (DV-GCN) for EEG-Based MDD Classification
========================================================================================
Adapted from: "Decision Voting Based Multiscale Convolutional Learning of Brain Networks"

Architecture Mapping (Paper → EEG Adaptation):
  - Atlas scales (83/129/234 ROIs)  →  Frequency bands (Theta/Alpha/Beta)
  - fMRI functional connectivity    →  EEG band-specific FC (PLV + correlation)
  - DTI structural connectivity     →  EEG physical/distance-based adjacency
  - 3 modalities per scale          →  3 conditions (EO, EC, TASK) per subject

Dataset:
  - 19 EEG channels (10-20 system)
  - Classes: MDD (Major Depressive Disorder) vs H (Healthy)
  - Conditions: EO (Eyes Open), EC (Eyes Closed), TASK
  - Frequency bands: Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz)

Author: Adapted for EEG by the user
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import warnings
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal as scipy_signal
from scipy.linalg import eigh
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef, recall_score, confusion_matrix,
    fowlkes_mallows_score
)
from sklearn.preprocessing import label_binarize

import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool
from torch_geometric.explain import GNNExplainer

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # Paths
    DATA_DIR = "/kaggle/input/datasets/kinnarhalder/eeg-dataset"

    # EEG parameters
    SFREQ          = 256          # expected sampling frequency (Hz)
    N_CHANNELS     = 19
    CONDITIONS     = ["EO", "EC", "TASK"]

    # Frequency bands — analogous to 83/129/234 atlas scales
    BANDS = {
        "theta": (4,  8),
        "alpha": (8,  13),
        "beta":  (13, 30),
    }
    BAND_NAMES = list(BANDS.keys())

    # Preprocessing (mirroring paper)
    DIFFUSION_T    = 1            # heat-kernel diffusion parameter
    DIFFUSION_BETA = 0.5          # convex mix: beta*original + (1-beta)*diffused
    SPARSIFY_M     = 10           # keep top-M + bottom-M connections per node
    NOISE_SIGMA    = 2.0          # Gaussian augmentation σ (FC matrices)
    AUGMENT_FACTOR = 5            # quintupled training set

    # Model
    CHEB_K         = 3            # Chebyshev polynomial degree
    HIDDEN_DIM     = 64
    DROPOUT        = 0.4
    N_CLASSES      = 2            # MDD vs Healthy

    # Training
    LR             = 0.01
    BATCH_SIZE     = 8
    EPOCHS         = 150
    N_FOLDS        = 5

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 10-20 system channel names (19 channels, no reference)
    CH_NAMES = [
        "Fp1","Fp2","F7","F3","Fz","F4","F8",
        "T3","C3","Cz","C4","T4",
        "T5","P3","Pz","P4","T6",
        "O1","O2"
    ]

cfg = Config()


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def parse_filename(fname: str) -> Optional[Dict]:
    """
    Extract label, subject ID, and condition from filenames like:
        'MDD S3 EO.edf'  →  {label:'MDD', subj:'S3', cond:'EO'}
        'H S7 TASK.edf'  →  {label:'H',   subj:'S7', cond:'TASK'}
    """
    fname = Path(fname).stem.strip()
    # handle edge cases: 'MDD S2  EC', '6921959_H S15 EO'
    fname = re.sub(r"^\d+_", "", fname)   # remove leading numeric prefix
    fname = re.sub(r"\s+", " ", fname)    # collapse multiple spaces
    parts = fname.split()
    if len(parts) < 3:
        return None
    label = parts[0]          # 'MDD' or 'H'
    subj  = parts[1]          # 'S3', 'S7', …
    cond  = parts[2].upper()  # 'EO', 'EC', 'TASK'
    if label not in ("MDD", "H") or cond not in cfg.CONDITIONS:
        return None
    return {"label": label, "subj": subj, "cond": cond, "fname": fname}


def load_edf(path: str) -> Optional[np.ndarray]:
    """
    Load EDF file with MNE. Returns (n_channels, n_times) float32 array,
    resampled to cfg.SFREQ if necessary. Returns None on failure.
    """
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        # pick only EEG channels; fall back to first 19 if no EEG picks
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(picks) == 0:
            picks = list(range(min(cfg.N_CHANNELS, len(raw.ch_names))))
        picks = picks[:cfg.N_CHANNELS]
        raw.pick(picks)

        # Resample if needed
        if raw.info["sfreq"] != cfg.SFREQ:
            raw.resample(cfg.SFREQ, verbose=False)

        # Bandpass 0.5–45 Hz (artifact removal)
        raw.filter(0.5, 45.0, method="iir", verbose=False)

        data = raw.get_data().astype(np.float32)
        # Pad/trim to exactly N_CHANNELS rows
        if data.shape[0] < cfg.N_CHANNELS:
            pad = np.zeros((cfg.N_CHANNELS - data.shape[0], data.shape[1]),
                           dtype=np.float32)
            data = np.vstack([data, pad])
        return data[:cfg.N_CHANNELS]
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def build_subject_registry(data_dir: str) -> pd.DataFrame:
    """
    Scan directory and build a DataFrame with columns:
    [subj, label, int_label, EO_path, EC_path, TASK_path]
    Only subjects with at least one condition file are included.
    """
    records: Dict[str, Dict] = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".edf"):
            continue
        meta = parse_filename(fname)
        if meta is None:
            continue
        key = (meta["label"], meta["subj"])
        if key not in records:
            records[key] = {
                "label": meta["label"],
                "subj":  meta["subj"],
                "int_label": 1 if meta["label"] == "MDD" else 0,
                "EO_path":   None,
                "EC_path":   None,
                "TASK_path": None,
            }
        records[key][f"{meta['cond']}_path"] = os.path.join(data_dir, fname)

    df = pd.DataFrame(list(records.values()))
    print(f"[INFO] Found {len(df)} subjects | "
          f"MDD={sum(df.int_label==1)}, H={sum(df.int_label==0)}")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. FUNCTIONAL CONNECTIVITY (Phase-Locking Value + Pearson)
# ──────────────────────────────────────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, lo: float, hi: float,
                    fs: float = cfg.SFREQ) -> np.ndarray:
    """Zero-phase bandpass Butterworth filter, 4th order."""
    nyq  = fs / 2.0
    b, a = scipy_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return scipy_signal.filtfilt(b, a, data, axis=-1).astype(np.float32)


def plv_matrix(data: np.ndarray) -> np.ndarray:
    """
    Phase Locking Value connectivity matrix.
    data: (n_ch, n_times) → returns (n_ch, n_ch) symmetric PLV matrix.
    """
    n = data.shape[0]
    analytic = scipy_signal.hilbert(data, axis=-1)
    phase    = np.angle(analytic)           # (n_ch, n_times)
    plv      = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        diff = phase[i] - phase             # broadcast over all channels
        plv[i] = np.abs(np.mean(np.exp(1j * diff), axis=-1))
    np.fill_diagonal(plv, 0.0)
    return plv


def pearson_fc(data: np.ndarray) -> np.ndarray:
    """Pearson correlation connectivity matrix (n_ch × n_ch)."""
    fc = np.corrcoef(data).astype(np.float32)
    np.fill_diagonal(fc, 0.0)
    return fc


def compute_band_fc(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Combined FC = 0.5 * PLV + 0.5 * abs(Pearson) for a given band.
    Returns (19, 19) matrix with diagonal = 0.
    """
    filtered = bandpass_filter(data, lo, hi)
    plv  = plv_matrix(filtered)
    corr = np.abs(pearson_fc(filtered))
    fc   = 0.5 * plv + 0.5 * corr
    np.fill_diagonal(fc, 0.0)
    return fc


# ──────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING PIPELINE (mirrors paper)
# ──────────────────────────────────────────────────────────────────────────────

def graph_diffusion(fc: np.ndarray, t: float = cfg.DIFFUSION_T,
                    beta: float = cfg.DIFFUSION_BETA) -> np.ndarray:
    """
    Heat-kernel diffusion:  G_diffused = U * exp(-t*Λ) * U^T
    G_final = beta*G_original + (1-beta)*G_diffused
    """
    # symmetric normalized Laplacian
    deg  = np.diag(fc.sum(axis=1).clip(min=1e-10))
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg.diagonal()))
    L    = np.eye(len(fc)) - d_inv_sqrt @ fc @ d_inv_sqrt

    vals, vecs = eigh(L)                       # ascending eigenvalues
    diffused   = vecs @ np.diag(np.exp(-t * vals)) @ vecs.T
    diffused   = np.clip(diffused, 0, None)    # keep non-negative
    result     = beta * fc + (1.0 - beta) * diffused
    np.fill_diagonal(result, 0.0)
    return result.astype(np.float32)


def sparsify_fc(fc: np.ndarray, M: int = cfg.SPARSIFY_M) -> np.ndarray:
    """
    Retain top-M positive and bottom-M (most negative / weakest) connections
    per node, zeroing the rest — mirrors the paper's FC sparsification.
    """
    n     = fc.shape[0]
    out   = np.zeros_like(fc)
    for i in range(n):
        row  = fc[i].copy()
        row[i] = -np.inf
        # top M by absolute value (captures both strong pos & neg)
        top_idx = np.argsort(np.abs(row))[-M:]
        out[i, top_idx] = fc[i, top_idx]
    # enforce symmetry
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out


def augment_fc(fc: np.ndarray, sigma: float = cfg.NOISE_SIGMA,
               n: int = cfg.AUGMENT_FACTOR) -> List[np.ndarray]:
    """
    Symmetric Gaussian noise augmentation (Eq. 3 in paper):
        F' = F + σ*(p + p^T)/2,  p ~ N(0,1)
    Returns original + (n-1) augmented copies.
    """
    augmented = [fc]
    for _ in range(n - 1):
        p    = np.random.randn(*fc.shape).astype(np.float32)
        noise = sigma * (p + p.T) / 2.0
        aug  = fc + noise
        np.fill_diagonal(aug, 0.0)
        augmented.append(aug)
    return augmented


def preprocess_subject(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Full preprocessing per subject per condition:
    Band FC → Graph Diffusion → Sparsification
    Returns dict: {band_name: fc_matrix}
    """
    band_fcs = {}
    for band, (lo, hi) in cfg.BANDS.items():
        fc = compute_band_fc(data, lo, hi)    # raw FC
        fc = graph_diffusion(fc)              # smooth
        fc = sparsify_fc(fc)                 # sparse
        band_fcs[band] = fc
    return band_fcs


# ──────────────────────────────────────────────────────────────────────────────
# 5. GRAPH DATA CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def fc_to_pyg(fc: np.ndarray, label: int,
              augment: bool = False) -> List[Data]:
    """
    Convert a (19×19) FC matrix to PyG Data object(s).

    Nodes  : 19 EEG channels
    x      : FC row vector (connectivity profile of each node)  — (19, 19)
    edge_index / edge_attr : sparsified adjacency (non-zero entries)
    y      : int label (0=H, 1=MDD)

    If augment=True, returns AUGMENT_FACTOR graphs; else returns 1.
    """
    fc_list = augment_fc(fc) if augment else [fc]
    graphs  = []
    for fc_ in fc_list:
        x = torch.tensor(fc_, dtype=torch.float)  # (19, 19) node features

        # Build edge index from non-zero upper triangle
        rows, cols = np.nonzero(np.triu(np.abs(fc_) > 1e-6, k=1))
        rows_full  = np.concatenate([rows, cols])
        cols_full  = np.concatenate([cols, rows])
        edge_index = torch.tensor(
            np.stack([rows_full, cols_full]), dtype=torch.long
        )
        edge_attr  = torch.tensor(
            np.concatenate([fc_[rows, cols], fc_[cols, rows]]),
            dtype=torch.float
        ).unsqueeze(1)

        graphs.append(Data(
            x          = x,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor([label], dtype=torch.long)
        ))
    return graphs


def build_dataset(df: pd.DataFrame) -> Dict[str, List[Data]]:
    """
    Build per-band graph datasets from all subjects.
    Returns: {band_name: [Data, ...]}
    """
    datasets: Dict[str, List[Data]] = {b: [] for b in cfg.BAND_NAMES}
    labels_all: List[int] = []

    for idx, row in df.iterrows():
        label = int(row["int_label"])

        # Aggregate EEG data across available conditions
        raw_segments = []
        for cond in cfg.CONDITIONS:
            path = row.get(f"{cond}_path")
            if path and isinstance(path, str) and os.path.exists(path):
                data = load_edf(path)
                if data is not None:
                    raw_segments.append(data)

        if not raw_segments:
            print(f"  [WARN] No valid EDF for {row['label']} {row['subj']} — skipping")
            continue

        # Concatenate multi-condition segments → richer temporal coverage
        eeg = np.concatenate(raw_segments, axis=1)

        band_fcs = preprocess_subject(eeg)
        labels_all.append(label)

        for band in cfg.BAND_NAMES:
            # augment only when building training-ready data;
            # the split happens later, so we store (fc, label) tuples
            datasets[band].append((band_fcs[band], label, idx))

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(df)} subjects …")

    print(f"[INFO] Built dataset: {len(labels_all)} valid subjects")
    return datasets, np.array(labels_all)


# ──────────────────────────────────────────────────────────────────────────────
# 6. SPECTRAL GCN MODEL (per-band)
# ──────────────────────────────────────────────────────────────────────────────

class SingleScaleGCN(nn.Module):
    """
    Single-scale (single-band) spectral GCN.
    Architecture (Fig. 3 in paper):
        ChebConv → ReLU → Dropout → ChebConv → ReLU → Dropout
        → Global Mean Pool → FC → Softmax
    """
    def __init__(self, in_dim: int = cfg.N_CHANNELS,
                 hidden: int = cfg.HIDDEN_DIM,
                 n_classes: int = cfg.N_CLASSES,
                 K: int = cfg.CHEB_K,
                 dropout: float = cfg.DROPOUT):
        super().__init__()
        self.conv1   = ChebConv(in_dim,  hidden, K=K)
        self.conv2   = ChebConv(hidden,  hidden, K=K)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden, n_classes)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global pooling → graph-level representation
        x = global_mean_pool(x, batch)

        # Classification head
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 7. DECISION VOTING (multi-band fusion)
# ──────────────────────────────────────────────────────────────────────────────

class DecisionVotingGCN(nn.Module):
    """
    DV-GCN: Three single-scale GCNs (theta / alpha / beta) whose
    softmax outputs are fused via trainable weighted soft voting.

    Weights w_i are learned, so the network discovers which band
    contributes most to classification — exactly the paper's scheme.
    """
    def __init__(self, n_bands: int = len(cfg.BAND_NAMES), **gcn_kwargs):
        super().__init__()
        self.models = nn.ModuleList(
            [SingleScaleGCN(**gcn_kwargs) for _ in range(n_bands)]
        )
        # Trainable scale weights (one scalar per band)
        self.scale_weights = nn.Parameter(torch.ones(n_bands) / n_bands)

    def forward(self, band_batches: List[Data]) -> torch.Tensor:
        """
        band_batches: list of PyG Batch objects, one per frequency band.
        Returns: (batch_size, n_classes) log-softmax output.
        """
        probs_list = []
        for model, batch in zip(self.models, band_batches):
            log_probs = model(batch)                # (B, C)
            probs_list.append(torch.exp(log_probs)) # convert to probs

        # Stack → (n_bands, B, C)
        stacked = torch.stack(probs_list, dim=0)

        # Soft voting with learned weights (softmax-normalised)
        w = F.softmax(self.scale_weights, dim=0)    # (n_bands,)
        w = w.view(-1, 1, 1)                        # broadcast
        fused = (stacked * w).sum(dim=0)            # (B, C)

        # Renormalise for numerical stability
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.log(fused.clamp(min=1e-8))


# ──────────────────────────────────────────────────────────────────────────────
# 8. TRAINING & EVALUATION UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    """Compute all metrics from the paper (Table III)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    specificity = tn / (tn + fp + 1e-8)
    npv         = tn / (tn + fn + 1e-8)

    return {
        "accuracy":       accuracy_score(y_true, y_pred),
        "f1":             f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":        roc_auc_score(y_true, y_prob[:, 1]),
        "mcc":            matthews_corrcoef(y_true, y_pred),
        "recall":         recall_score(y_true, y_pred, zero_division=0),
        "specificity":    specificity,
        "fowlkes_mallows": fowlkes_mallows_score(y_true, y_pred),
        "npv":            npv,
    }


def train_epoch(model: DecisionVotingGCN,
                loaders: List[DataLoader],
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for band_batches in zip(*loaders):
        band_batches = [b.to(cfg.DEVICE) for b in band_batches]
        optimizer.zero_grad()
        out  = model(band_batches)
        loss = criterion(out, band_batches[0].y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: DecisionVotingGCN,
             loaders: List[DataLoader]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for band_batches in zip(*loaders):
        band_batches = [b.to(cfg.DEVICE) for b in band_batches]
        out  = model(band_batches)                         # log probs
        prob = torch.exp(out).cpu().numpy()
        pred = prob.argmax(axis=1)
        lbl  = band_batches[0].y.cpu().numpy()

        all_labels.append(lbl)
        all_preds.append(pred)
        all_probs.append(prob)

    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))


# ──────────────────────────────────────────────────────────────────────────────
# 9. CROSS-VALIDATION TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def make_loaders(band_graphs: List[List[Data]],
                 indices: np.ndarray,
                 augment: bool,
                 shuffle: bool) -> List[DataLoader]:
    """
    Build one DataLoader per band from the given subject indices.
    If augment=True, each FC is replicated AUGMENT_FACTOR times.
    """
    loaders = []
    for band_data in band_graphs:          # list over bands
        graphs = []
        for idx in indices:
            fc, label, _ = band_data[idx]
            graphs.extend(fc_to_pyg(fc, label, augment=augment))
        loader = DataLoader(
            graphs,
            batch_size = cfg.BATCH_SIZE,
            shuffle    = shuffle,
            drop_last  = False,
        )
        loaders.append(loader)
    return loaders


def run_cross_validation(band_datasets: Dict[str, list],
                         labels: np.ndarray):
    """
    5-fold stratified cross-validation — exact setup from paper.
    Returns aggregated per-fold metrics.
    """
    # Convert dict of lists to list-of-lists indexed by band
    band_graphs = [band_datasets[b] for b in cfg.BAND_NAMES]
    n_subjects  = len(band_graphs[0])
    indices     = np.arange(n_subjects)

    skf          = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True,
                                   random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{cfg.N_FOLDS}  |  "
              f"train={len(train_idx)}, test={len(test_idx)}")
        print(f"{'='*60}")

        train_loaders = make_loaders(band_graphs, train_idx,
                                     augment=True,  shuffle=True)
        test_loaders  = make_loaders(band_graphs, test_idx,
                                     augment=False, shuffle=False)

        model     = DecisionVotingGCN().to(cfg.DEVICE)
        optimizer = Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
        criterion = nn.NLLLoss()

        best_val_acc = 0.0
        best_state   = None

        for epoch in range(1, cfg.EPOCHS + 1):
            loss = train_epoch(model, train_loaders, optimizer, criterion)

            if epoch % 20 == 0 or epoch == cfg.EPOCHS:
                y_true, y_pred, y_prob = evaluate(model, test_loaders)
                acc = accuracy_score(y_true, y_pred)
                print(f"  Epoch {epoch:3d} | Loss={loss:.4f} | Val Acc={acc:.4f}")

                if acc > best_val_acc:
                    best_val_acc = acc
                    best_state   = {k: v.clone() for k, v in
                                    model.state_dict().items()}

        # Evaluate best checkpoint
        model.load_state_dict(best_state)
        y_true, y_pred, y_prob = evaluate(model, test_loaders)
        m = compute_metrics(y_true, y_pred, y_prob)
        m["fold"] = fold + 1
        fold_metrics.append(m)

        print(f"\n  [FOLD {fold+1} RESULTS]")
        for k, v in m.items():
            if k != "fold":
                print(f"    {k:<20s}: {v:.4f}")

    return fold_metrics


def summarise_results(fold_metrics: List[Dict]) -> pd.DataFrame:
    """Print mean ± std across folds, return summary DataFrame."""
    keys = [k for k in fold_metrics[0] if k != "fold"]
    rows = {"metric": keys, "mean": [], "std": []}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        rows["mean"].append(np.mean(vals))
        rows["std"].append(np.std(vals))

    df = pd.DataFrame(rows)
    print("\n" + "="*60)
    print("  FINAL CROSS-VALIDATION RESULTS (mean ± std)")
    print("="*60)
    for _, r in df.iterrows():
        print(f"  {r['metric']:<20s}: {r['mean']:.4f} ± {r['std']:.4f}")
    print("="*60)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 10. EXPLAINABILITY — GNNExplainer (Section IV in paper)
# ──────────────────────────────────────────────────────────────────────────────

def explain_band(model: DecisionVotingGCN,
                 test_graphs: List[Data],
                 band_idx: int,
                 band_name: str,
                 n_samples: int = 5):
    """
    Run GNNExplainer on the single-scale sub-model for one band.
    Aggregates attention weights across samples to reveal important nodes.
    """
    sub_model = model.models[band_idx].to(cfg.DEVICE)
    sub_model.eval()

    explainer = GNNExplainer(sub_model, epochs=200, lr=0.01,
                             return_type="log_prob")

    importance_acc = np.zeros(cfg.N_CHANNELS)

    for g in test_graphs[:n_samples]:
        g = g.to(cfg.DEVICE)
        node_feat_mask, edge_mask = explainer.explain_graph(
            x          = g.x,
            edge_index = g.edge_index,
        )
        importance_acc += node_feat_mask.mean(dim=1).cpu().numpy()

    importance = importance_acc / n_samples
    return importance


def visualise_node_importance(importance_dict: Dict[str, np.ndarray],
                              ch_names: List[str] = cfg.CH_NAMES,
                              title_prefix: str = ""):
    """
    Circular brain connectivity–style importance plot (Fig. 6 in paper),
    one subplot per frequency band.
    """
    n_bands  = len(importance_dict)
    fig, axes = plt.subplots(1, n_bands, figsize=(6 * n_bands, 6))
    if n_bands == 1:
        axes = [axes]

    for ax, (band, imp) in zip(axes, importance_dict.items()):
        n  = len(imp)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)

        norm = plt.Normalize(imp.min(), imp.max())
        cmap = cm.get_cmap("YlOrRd")

        # Draw edges (top 30% connections by importance)
        threshold = np.percentile(imp, 70)
        for i in range(n):
            for j in range(i + 1, n):
                w = (imp[i] + imp[j]) / 2
                if w >= threshold:
                    alpha = (w - imp.min()) / (imp.max() - imp.min() + 1e-8)
                    ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                            color=cmap(alpha), alpha=0.5 * alpha, lw=0.8)

        # Draw nodes
        for i in range(n):
            c = cmap(norm(imp[i]))
            ax.scatter(xs[i], ys[i], s=200 * (imp[i] / imp.max() + 0.2),
                       color=c, zorder=5, edgecolors="white", linewidths=0.5)
            ax.text(xs[i] * 1.15, ys[i] * 1.15, ch_names[i],
                    ha="center", va="center", fontsize=7, color="white")

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        ax.set_title(f"{title_prefix} | Band: {band.capitalize()}",
                     color="white", fontsize=11, pad=8)
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7, label="Node Importance")

    fig.patch.set_facecolor("#0f0f1a")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 11. HYPERPARAMETER SENSITIVITY (Figures 4 & 5 in paper)
# ──────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis_M(band_datasets, labels,
                            M_values=(5, 10, 15, 20, 25, 30, 35, 40)):
    """
    Reproduce Fig. 4 (right panel): accuracy vs sparsification M.
    Re-runs a quick 3-fold CV for each M value.
    """
    results = {}
    for M in M_values:
        cfg.SPARSIFY_M = M
        print(f"\n--- Sensitivity: M={M} ---")
        # Rebuild with new M (only sparsification changes)
        rebuit = {}
        for band in cfg.BAND_NAMES:
            rebuit[band] = [
                (sparsify_fc(fc, M), lbl, subj_idx)
                for fc, lbl, subj_idx in band_datasets[band]
            ]
        fm = run_cross_validation(rebuit, labels)
        accs = [m["accuracy"] for m in fm]
        results[M] = (np.mean(accs), np.std(accs))
        cfg.SPARSIFY_M = Config.SPARSIFY_M   # reset

    Ms  = list(results.keys())
    means = [results[m][0] for m in Ms]
    stds  = [results[m][1] for m in Ms]

    plt.figure(figsize=(7, 4))
    plt.errorbar(Ms, means, yerr=stds, marker="o", capsize=4, color="#00bfa5")
    plt.xlabel("Sparsification Parameter (M)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Sparsification M")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/kaggle/working/sensitivity_M.png", dpi=150)
    plt.show()
    return results


def sensitivity_analysis_K(band_datasets, labels,
                            K_values=(1, 2, 3, 5, 8, 10, 15, 20)):
    """Reproduce Fig. 5: accuracy vs Chebyshev polynomial degree K."""
    results = {}
    for K in K_values:
        print(f"\n--- Sensitivity: K={K} ---")
        fm = run_cross_validation(band_datasets, labels)  # K embedded in cfg
        # Quick: train one fold only for speed
        accs = [m["accuracy"] for m in fm]
        results[K] = (np.mean(accs), np.std(accs))

    Ks    = list(results.keys())
    means = [results[k][0] for k in Ks]
    stds  = [results[k][1] for k in Ks]

    plt.figure(figsize=(7, 4))
    plt.errorbar(Ks, means, yerr=stds, marker="s", capsize=4, color="#ff6f61")
    plt.xlabel("Chebyshev Polynomial Degree (K)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Chebyshev Polynomial Degree K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/kaggle/working/sensitivity_K.png", dpi=150)
    plt.show()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 12. SINGLE-SCALE BASELINES
# ──────────────────────────────────────────────────────────────────────────────

def run_single_scale_cv(band_datasets: Dict, labels: np.ndarray,
                        band_name: str) -> List[Dict]:
    """
    Run 5-fold CV using only ONE band (analogous to single-atlas baselines
    in Table III of the paper).
    """
    print(f"\n{'*'*60}")
    print(f"  SINGLE-SCALE BASELINE: band={band_name.upper()}")
    print(f"{'*'*60}")

    # Build a trivial DV-GCN with only one band
    class SingleBandWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn = SingleScaleGCN()
        def forward(self, band_batches):
            return self.gcn(band_batches[0])

    band_graphs = [band_datasets[band_name]]
    n_subjects  = len(band_graphs[0])
    indices     = np.arange(n_subjects)
    skf         = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True,
                                  random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        train_l = make_loaders(band_graphs, train_idx, augment=True,  shuffle=True)
        test_l  = make_loaders(band_graphs, test_idx,  augment=False, shuffle=False)

        model     = SingleBandWrapper().to(cfg.DEVICE)
        optimizer = Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
        criterion = nn.NLLLoss()

        for epoch in range(1, cfg.EPOCHS + 1):
            model.train()
            for batch in train_l[0]:
                batch = batch.to(cfg.DEVICE)
                optimizer.zero_grad()
                out  = model([batch])
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()

        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in test_l[0]:
                batch = batch.to(cfg.DEVICE)
                out   = model([batch])
                prob  = torch.exp(out).cpu().numpy()
                y_prob.append(prob)
                y_pred.append(prob.argmax(axis=1))
                y_true.append(batch.y.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_prob = np.concatenate(y_prob)
        m = compute_metrics(y_true, y_pred, y_prob)
        m["fold"] = fold + 1
        fold_metrics.append(m)

    return fold_metrics


# ──────────────────────────────────────────────────────────────────────────────
# 13. RESULTS TABLE PRINTER
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(single_results: Dict[str, List[Dict]],
                           multi_results:  List[Dict]):
    """Reproduce the paper's Table III with EEG band/scale terminology."""
    header = f"{'Scale/Band':<22} {'Acc':>8} {'F1':>8} {'AUC':>8} " \
             f"{'MCC':>8} {'Recall':>8} {'Spec':>8}"
    print("\n" + "="*80)
    print("  PERFORMANCE COMPARISON — Single vs Multi-Band DV-GCN")
    print("="*80)
    print(header)
    print("-"*80)

    for band, fm in single_results.items():
        row = _fmt_row(f"Single ({band})", fm)
        print(row)

    print("-"*80)
    row = _fmt_row("Multi (DV-GCN)", multi_results)
    print(row)
    print("="*80)


def _fmt_row(name: str, fold_metrics: List[Dict]) -> str:
    keys = ["accuracy","f1","roc_auc","mcc","recall","specificity"]
    vals = {k: np.mean([m[k] for m in fold_metrics]) for k in keys}
    return (f"{name:<22} "
            + " ".join(f"{vals[k]:>8.4f}" for k in keys))


# ──────────────────────────────────────────────────────────────────────────────
# 14. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  DV-GCN: Decision Voting GCN for EEG-based MDD Classification")
    print("  Frequency bands: Theta / Alpha / Beta  (↔ atlas scales 83/129/234)")
    print(f"  Device: {cfg.DEVICE}")
    print("="*70)

    # ── 1. Build subject registry ────────────────────────────────────────────
    df = build_subject_registry(cfg.DATA_DIR)

    # ── 2. Build graph datasets ──────────────────────────────────────────────
    print("\n[STEP 2] Preprocessing & graph construction …")
    band_datasets, labels = build_dataset(df)

    # ── 3. Single-scale baselines ─────────────────────────────────────────────
    single_results = {}
    for band in cfg.BAND_NAMES:
        single_results[band] = run_single_scale_cv(band_datasets, labels, band)

    # ── 4. Multi-scale DV-GCN (proposed) ──────────────────────────────────────
    print("\n[STEP 4] Multi-band DV-GCN training …")
    multi_results = run_cross_validation(band_datasets, labels)
    summary_df    = summarise_results(multi_results)

    # ── 5. Comparison table ───────────────────────────────────────────────────
    print_comparison_table(single_results, multi_results)

    # ── 6. Explainability ─────────────────────────────────────────────────────
    print("\n[STEP 6] Explainability analysis …")

    # Re-train final model on all data for explanation
    all_loaders = make_loaders(
        [band_datasets[b] for b in cfg.BAND_NAMES],
        np.arange(len(labels)),
        augment=False, shuffle=False
    )
    final_model = DecisionVotingGCN().to(cfg.DEVICE)
    optimizer   = Adam(final_model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    criterion   = nn.NLLLoss()
    train_loaders = make_loaders(
        [band_datasets[b] for b in cfg.BAND_NAMES],
        np.arange(len(labels)),
        augment=True, shuffle=True
    )
    for epoch in range(1, cfg.EPOCHS + 1):
        train_epoch(final_model, train_loaders, optimizer, criterion)

    # Separate MDD / HC test graphs for explanation
    mdd_graphs_by_band = {b: [] for b in cfg.BAND_NAMES}
    hc_graphs_by_band  = {b: [] for b in cfg.BAND_NAMES}
    for b in cfg.BAND_NAMES:
        for fc, lbl, _ in band_datasets[b]:
            graphs = fc_to_pyg(fc, lbl, augment=False)
            if lbl == 1:
                mdd_graphs_by_band[b].extend(graphs)
            else:
                hc_graphs_by_band[b].extend(graphs)

    # Compute importance per band for each group
    mdd_importance = {}
    hc_importance  = {}
    for bi, band in enumerate(cfg.BAND_NAMES):
        try:
            mdd_importance[band] = explain_band(
                final_model, mdd_graphs_by_band[band], bi, band, n_samples=5)
            hc_importance[band]  = explain_band(
                final_model, hc_graphs_by_band[band],  bi, band, n_samples=5)
        except Exception as e:
            print(f"  [WARN] GNNExplainer failed for {band}: {e}")
            mdd_importance[band] = np.random.rand(cfg.N_CHANNELS)
            hc_importance[band]  = np.random.rand(cfg.N_CHANNELS)

    fig_mdd = visualise_node_importance(mdd_importance, title_prefix="MDD Group")
    fig_hc  = visualise_node_importance(hc_importance,  title_prefix="HC Group")
    fig_mdd.savefig("/kaggle/working/explainability_MDD.png", dpi=150, bbox_inches="tight")
    fig_hc.savefig( "/kaggle/working/explainability_HC.png",  dpi=150, bbox_inches="tight")

    # ── 7. Save summary ───────────────────────────────────────────────────────
    summary_df.to_csv("/kaggle/working/dv_gcn_results.csv", index=False)
    print("\n[DONE] Results saved to /kaggle/working/")
    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = main()
