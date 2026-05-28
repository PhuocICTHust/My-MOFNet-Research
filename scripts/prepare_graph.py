"""
prepare_graph.py — MOFNet-Pro

Data context:
  mRNA  : 671 × 5000  → reduce to topk_mrna  (default 1000)
  miRNA : 671 × 200   → NO reduction (already small)
  Methy : 671 × 5000  → reduce to topk_methy (default 1000)
  CNV   : 671 × 5000  → reduce to topk_cnv   (default 1000)
  Labels: 5 classes [0=353, 1=42, 2=132, 3=31, 4=113] — severe imbalance

Fixes vs previous version:
  1. Remove double scaling (representation distortion bug)
  2. fill_diagonal(sim, -1) instead of 0 — ensures no self-loop
  3. StandardScaler fit on train only → transform all (no leakage)
  4. miRNA not reduced (200 features already small enough)
  5. FIX: f-string double-brace bug in NaN imputation print
     {{name}} and {{nan_count}} printed literal {name}/{nan_count} text
     instead of the actual variable values.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.data import Data


# ─────────────────────────────────────────────
# 1. ARGPARSE
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cancer",      required=True, choices=["BRCA", "COAD", "GBM", "OV"])
parser.add_argument("--seed",        type=int,   default=777)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--k",           type=int,   default=7,
                    help="KNN neighbors. k=7 balances connectivity and noise for N=671.")
parser.add_argument("--topk_mrna",   type=int,   default=1000)
parser.add_argument("--topk_methy",  type=int,   default=1000)
parser.add_argument("--topk_cnv",    type=int,   default=1000)
args = parser.parse_args()

repo_path   = r"E:\Cancer-classification-dataset"
cancer_type = args.cancer
base_path   = os.path.join(repo_path, cancer_type)


# ─────────────────────────────────────────────
# 2. SEED
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(args.seed)


# ─────────────────────────────────────────────
# 3. FEATURE SELECTION (MAD-based, fit on train)
# ─────────────────────────────────────────────
def select_top_mad(X: np.ndarray, train_idx: np.ndarray, k: int) -> np.ndarray:
    """
    Select k features with highest MAD, computed on TRAIN ONLY.
    MAD is more robust than variance with outliers — good for CNV (0/1/2)
    and Methy (bimodal).
    """
    X_train = X[train_idx]
    mad     = np.median(np.abs(X_train - np.median(X_train, axis=0)), axis=0)
    top_idx = np.argsort(mad)[-k:]
    return X[:, top_idx]


# ─────────────────────────────────────────────
# 4. BUILD KNN GRAPH
# ─────────────────────────────────────────────
def build_knn_graph(X: np.ndarray, k: int) -> torch.Tensor:
    """
    Mutual KNN graph from cosine similarity.

    Mutual KNN: keep edge(i,j) only if j in topk(i) AND i in topk(j).
    Effect:
      - Removes noisy one-directional edges
      - Cleaner graph with less structural bias than standard KNN
      - GATConv learns actual neighborhood, not similarity artifacts

    fill_diagonal(-1): absolutely prevents self-loops.
    """
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -1)

    sim_t       = torch.tensor(sim, dtype=torch.float32)
    _, topk_idx = torch.topk(sim_t, k=k, dim=1)          # [N, k]

    N = X.shape[0]

    # Build boolean adjacency matrix from KNN
    adj = torch.zeros(N, N, dtype=torch.bool)
    row = torch.arange(N).unsqueeze(1).expand(-1, k)     # [N, k]
    adj[row.reshape(-1), topk_idx.reshape(-1)] = True

    # Mutual: keep only edges that exist in both directions
    mutual = adj & adj.T                                  # [N, N]

    edge_index = mutual.nonzero(as_tuple=False).T         # [2, E]

    return edge_index


# ─────────────────────────────────────────────
# 5. LOAD DATA
# ─────────────────────────────────────────────
def load_omic(filename: str, fillna=None) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(base_path, filename), index_col=0).T
    df.index = df.index.astype(str).str[:16]
    if fillna is not None:
        df = df.fillna(fillna)
    return df

print(f"\n[1/5] Loading omics data for {cancer_type}...")
df_mrna  = load_omic(f'{cancer_type}_mRNA.csv')
df_mirna = load_omic(f'{cancer_type}_miRNA.csv', fillna=0)
df_methy = load_omic(f'{cancer_type}_Methy.csv')
df_cnv   = load_omic(f'{cancer_type}_CNV.csv')

common = sorted(
    set(df_mrna.index) & set(df_mirna.index) & set(df_methy.index) & set(df_cnv.index)
)
print(f"  → Common patients: {len(common)}")


# ─────────────────────────────────────────────
# 6. LABEL & SPLIT
# ─────────────────────────────────────────────
print("[2/5] Loading labels & splitting...")

labels_df = pd.read_csv(os.path.join(base_path, f'{cancer_type}_label.csv'))
if labels_df.shape[1] > 10:
    labels_df = labels_df.T
labels_df.index = df_mrna.index

y = LabelEncoder().fit_transform(labels_df.loc[common].values.ravel())

idx = np.arange(len(common))
train_idx, test_idx = train_test_split(
    idx, train_size=args.train_ratio, stratify=y, random_state=args.seed
)

train_mask = torch.zeros(len(common), dtype=torch.bool)
test_mask  = torch.zeros(len(common), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx]   = True
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"  → Train: {train_mask.sum().item()}, Test: {test_mask.sum().item()}")
print(f"  → Class distribution (full): {np.bincount(y)}")


# ─────────────────────────────────────────────
# 7. PREPROCESS: SCALE → FEATURE SELECT (1 scaler only)
# ─────────────────────────────────────────────
print("[3/5] Preprocessing features...")

def preprocess(df: pd.DataFrame, topk, name: str) -> np.ndarray:
    """
    Pipeline (single scaler — no double normalization):
      0. Median imputation for NaN (fit on train only)
      1. Fit StandardScaler on TRAIN → transform all
      2. MAD feature selection fit on TRAIN (if topk specified)
    """
    X = df.loc[common].values.astype(np.float32)

    # Step 0: Impute NaN with train median (fit on train only)
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        # FIX Bug 4: was f"  {{name}}: found {{nan_count}} NaN values…"
        # Double-braces in f-strings print literal {name} text, not the variable.
        print(f"  {name}: found {nan_count} NaN values → imputing with train median")
        train_median = np.nanmedian(X[train_idx], axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = train_median[j]

    # Step 1: Scale — fit on train only (no leakage)
    scaler = StandardScaler()
    X[train_idx] = scaler.fit_transform(X[train_idx])
    X[test_idx]  = scaler.transform(X[test_idx])

    # Step 2: Feature selection — computed on train only
    if topk is not None and topk < X.shape[1]:
        X = select_top_mad(X, train_idx, topk)
        print(f"  {name}: {df.shape[1]} → {X.shape[1]} features (MAD top-{topk})")
    else:
        print(f"  {name}: {X.shape[1]} features (no reduction)")

    return X

X_mrna  = preprocess(df_mrna,  args.topk_mrna,  "mRNA")
X_mirna = preprocess(df_mirna, None,             "miRNA")  # 200 features, no reduce
X_methy = preprocess(df_methy, args.topk_methy,  "Methy")
X_cnv   = preprocess(df_cnv,   args.topk_cnv,    "CNV")


# ─────────────────────────────────────────────
# 8. BUILD GRAPHS
# ─────────────────────────────────────────────
print(f"[4/5] Building KNN graphs (k={args.k})...")

def create_data(X: np.ndarray, name: str) -> Data:
    edge_index = build_knn_graph(X, args.k)
    print(f"  {name}: nodes={X.shape[0]}, edges={edge_index.shape[1]}, feats={X.shape[1]}")
    return Data(
        x          = torch.tensor(X, dtype=torch.float32),
        edge_index = edge_index,
        y          = y_tensor,
        train_mask = train_mask,
        test_mask  = test_mask,
    )

dataset = {
    "mRNA":  create_data(X_mrna,  "mRNA"),
    "miRNA": create_data(X_mirna, "miRNA"),
    "Methy": create_data(X_methy, "Methy"),
    "CNV":   create_data(X_cnv,   "CNV"),
}


# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
print("[5/5] Saving...")
save_path = os.path.join(repo_path, f"{cancer_type}_graph.pt")
torch.save(dataset, save_path)
print(f"\n✅ DONE! Saved to: {save_path}")