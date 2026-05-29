import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from torch_geometric.data import Data


# ─────────────────────────────────────────────
# 1. ARGPARSE
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cancer",      required=True,
                    choices=["BRCA", "COAD", "GBM", "LGG", "OV"],
                    help="Cancer type. Supported: BRCA (N=671, 5-class), "
                         "COAD (N=260, 4-class), GBM (N=244, 5-class), "
                         "LGG (N=247, 3-class), OV (N=284, 4-class).")
parser.add_argument("--seed",        type=int,   default=777)
parser.add_argument("--train_ratio", type=float, default=0.7,
                    help="Train split ratio. Default 0.7 (70/10/20 split).")
# FIX #1: added val_ratio for 70/10/20 split to eliminate data leakage in model selection.
# Previous 80/20 split: best checkpoint was selected by test F1 → implicit leakage.
# New 70/10/20: checkpoint selected by val F1, test set is untouched during training.
parser.add_argument("--val_ratio",   type=float, default=0.1,
                    help="Validation split ratio. Default 0.1 (70/10/20 split).")
parser.add_argument("--k",           type=int,   default=7,
                    help="Mutual KNN neighbors. k=7 works well for N≥200.")
parser.add_argument("--topk_mrna",   type=int,   default=1000)
parser.add_argument("--topk_mirna",  type=int,   default=None,
                    help="Top-k miRNA features by MAD (fit on train only). "
                         "Default=None → keep all features. "
                         "Recommended: ~100 for LGG/OV/GBM (high all-zero rates). "
                         "BRCA miRNA (366 features, 0%% zero) does not need reduction.")
parser.add_argument("--topk_methy",  type=int,   default=1000)
parser.add_argument("--topk_cnv",    type=int,   default=1000)
# FIX #2: PCA before KNN graph construction (optional, medium priority).
# PCA removes noise dimensions → cleaner graph topology → GATConv gets better neighborhood.
# Node features (Data.x) remain full-dimensional; PCA only affects WHICH patients are neighbors.
# Recommended: --pca_graph 64. Leave as None to skip PCA (backward-compatible).
parser.add_argument("--pca_graph",   type=int,   default=None,
                    help="If set, apply PCA to this many components before building KNN graph. "
                         "Cleaner topology, does not change node feature dimensions. "
                         "Recommended: 64. Default=None (no PCA).")
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

    MAD (Median Absolute Deviation):
        MAD(X) = median( |Xi - median(X)| )

    Why MAD instead of variance?
      - Uses median twice → robust against extreme outliers (CNV spikes, expression bursts).
      - All-zero features: median=0 → all deviations=0 → MAD=0 → never selected.
        This automatically filters uninformative all-zero miRNA features in LGG/GBM/OV
        without any special-casing.
      - Features that are 0 in SOME samples but non-zero in others → MAD > 0 → kept.
        These carry real biological signal (gene silencing patterns).
    """
    X_train = X[train_idx]
    mad     = np.median(np.abs(X_train - np.median(X_train, axis=0)), axis=0)
    top_idx = np.argsort(mad)[-k:]
    return X[:, top_idx]


# ─────────────────────────────────────────────
# 4. BUILD KNN GRAPH
# ─────────────────────────────────────────────
def build_knn_graph(X: np.ndarray, k: int, train_idx: np.ndarray,
                    pca_components: int = None) -> torch.Tensor:
    """
    Mutual KNN graph from cosine similarity.

    Mutual KNN: keep edge(i,j) only if j ∈ topk(i) AND i ∈ topk(j).
      - Removes noisy one-directional edges.
      - GATConv learns actual neighborhood structure.

    If pca_components is set:
      - PCA is fit on train_idx ONLY (no leakage).
      - Transform all patients for graph construction.
      - Node features (Data.x) are NOT reduced — only topology is affected.
    """
    X_graph = X.copy()

    if pca_components is not None and pca_components < X_graph.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        pca.fit(X_graph[train_idx])          # fit on train only
        X_graph = pca.transform(X_graph)
        explained = pca.explained_variance_ratio_.sum()
        print(f"    PCA: {X.shape[1]} → {X_graph.shape[1]} dims "
              f"(explained variance: {explained:.2%})")

    sim = cosine_similarity(X_graph)
    np.fill_diagonal(sim, -1)               # prevent self-loops

    sim_t       = torch.tensor(sim, dtype=torch.float32)
    _, topk_idx = torch.topk(sim_t, k=k, dim=1)    # [N, k]

    N   = X_graph.shape[0]
    adj = torch.zeros(N, N, dtype=torch.bool)
    row = torch.arange(N).unsqueeze(1).expand(-1, k)
    adj[row.reshape(-1), topk_idx.reshape(-1)] = True

    # Mutual: keep only edges present in both directions
    mutual     = adj & adj.T
    edge_index = mutual.nonzero(as_tuple=False).T    # [2, E]

    return edge_index


# ─────────────────────────────────────────────
# 5. LOAD DATA
# ─────────────────────────────────────────────
def load_omic(filename: str) -> pd.DataFrame:
    """
    Load omics CSV and transpose so rows = patients, columns = features.
    NaN handling is done downstream in preprocess() via train-median imputation
    for ALL omics (consistent, no leakage).
    """
    df = pd.read_csv(os.path.join(base_path, filename), index_col=0).T
    df.index = df.index.astype(str).str[:16]
    return df

print(f"\n[1/5] Loading omics data for {cancer_type}...")

df_mrna  = load_omic(f'{cancer_type}_mRNA.csv')
df_mirna = load_omic(f'{cancer_type}_miRNA.csv')   # FIX #3: removed fillna=0.
# NaN → train-median imputation in preprocess().
# Consistent with Methy handling.
# For LGG/GBM/OV: zeros are biological (not NaN),
# so no change for those cancers.
df_methy = load_omic(f'{cancer_type}_Methy.csv')
df_cnv   = load_omic(f'{cancer_type}_CNV.csv')

common = sorted(
    set(df_mrna.index) & set(df_mirna.index) & set(df_methy.index) & set(df_cnv.index)
)
print(f"  → Common patients: {len(common)}")


# ─────────────────────────────────────────────
# 6. LABEL & SPLIT (70 / 10 / 20)
# ─────────────────────────────────────────────
print("[2/5] Loading labels & splitting (70/10/20)...")

labels_df = pd.read_csv(os.path.join(base_path, f'{cancer_type}_label.csv'))
if labels_df.shape[1] > 10:
    labels_df = labels_df.T
labels_df.index = df_mrna.index

y = LabelEncoder().fit_transform(labels_df.loc[common].values.ravel())
idx = np.arange(len(common))

# Step 1: carve out test (20%)
test_size = 1.0 - args.train_ratio - args.val_ratio
train_val_idx, test_idx = train_test_split(
    idx, test_size=test_size, stratify=y, random_state=args.seed
)

# Step 2: split remainder into train (70%) and val (10%)
val_size_adjusted = args.val_ratio / (args.train_ratio + args.val_ratio)
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=val_size_adjusted,
    stratify=y[train_val_idx], random_state=args.seed
)

train_mask = torch.zeros(len(common), dtype=torch.bool)
val_mask   = torch.zeros(len(common), dtype=torch.bool)
test_mask  = torch.zeros(len(common), dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"  → Train: {train_mask.sum().item()}, "
      f"Val: {val_mask.sum().item()}, "
      f"Test: {test_mask.sum().item()}")
print(f"  → Class distribution (full): {np.bincount(y)}")

for cls_i, cnt in enumerate(np.bincount(y)):
    if cnt < 5:
        print(f"  ⚠  Class {cls_i} has only {cnt} total samples — "
              f"per-class evaluation will be unreliable.")


# ─────────────────────────────────────────────
# 7. PREPROCESS: IMPUTE → SCALE → FEATURE SELECT
# ─────────────────────────────────────────────
print("[3/5] Preprocessing features...")

def preprocess(df: pd.DataFrame, topk, name: str) -> np.ndarray:
    """
    Unified pipeline — no double normalization, no leakage:
      0. Train-median imputation for NaN (fit on train only)
      1. StandardScaler fit on TRAIN → transform train/val/test
      2. MAD feature selection fit on TRAIN (if topk specified)
    """
    X = df.loc[common].values.astype(np.float32)

    # Step 0: Impute NaN with train median (ALL omics, consistent)
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  {name}: {nan_count} NaN values → train-median imputation")
        train_median = np.nanmedian(X[train_idx], axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            if nan_mask.any():
                X[nan_mask, j] = train_median[j]

    # Step 1: Scale — fit on train only (zero-variance features → scale=1, stays 0)
    scaler = StandardScaler()
    X[train_idx] = scaler.fit_transform(X[train_idx])
    X[val_idx]   = scaler.transform(X[val_idx])      # FIX #4: was missing val
    X[test_idx]  = scaler.transform(X[test_idx])

    # Step 2: Feature selection — MAD computed on train only
    if topk is not None and topk < X.shape[1]:
        X = select_top_mad(X, train_idx, topk)
        print(f"  {name}: {df.shape[1]} → {X.shape[1]} features (MAD top-{topk})")
    else:
        print(f"  {name}: {X.shape[1]} features (no reduction)")

    return X

X_mrna  = preprocess(df_mrna,  args.topk_mrna,  "mRNA")
X_mirna = preprocess(df_mirna, args.topk_mirna,  "miRNA")
X_methy = preprocess(df_methy, args.topk_methy,  "Methy")
X_cnv   = preprocess(df_cnv,   args.topk_cnv,    "CNV")


# ─────────────────────────────────────────────
# 8. BUILD GRAPHS
# ─────────────────────────────────────────────
pca_info = f", PCA→{args.pca_graph}d" if args.pca_graph else ""
print(f"[4/5] Building mutual KNN graphs (k={args.k}{pca_info})...")

def create_data(X: np.ndarray, name: str) -> Data:
    edge_index = build_knn_graph(X, args.k, train_idx, args.pca_graph)
    print(f"  {name}: nodes={X.shape[0]}, edges={edge_index.shape[1]}, feats={X.shape[1]}")
    return Data(
        x          = torch.tensor(X, dtype=torch.float32),
        edge_index = edge_index,
        y          = y_tensor,
        train_mask = train_mask,
        val_mask   = val_mask,       # FIX #1: added val_mask
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
print(f"   num_classes for {cancer_type}: {int(y_tensor.max().item()) + 1}")
print(f"   Split: {train_mask.sum().item()} train / "
      f"{val_mask.sum().item()} val / "
      f"{test_mask.sum().item()} test")
print(f"   (pass num_classes to main.py via --num_classes, or leave 0 for auto-detect)")