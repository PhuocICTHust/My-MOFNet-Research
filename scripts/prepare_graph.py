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
parser.add_argument("--seed",        type=int,   default=777,
                    help="Global RNG seed (numpy/torch). For the data SPLIT seed "
                         "use --split_seed.")
# FIX P3: separate split seed → enables honest repeated stratified CV.
# The model-init seeds inside main.py only vary weight init / dropout; the
# DATA SPLIT is baked into the graph file. With a single fixed split, the
# 5 init-seeds understate true small-N variance. Regenerate the graph for
# split_seed ∈ {0..4}, run main.py on each, and report mean ± std of the
# 5 split-level test scores — that is the defensible variance for small N.
parser.add_argument("--split_seed",  type=int,   default=None,
                    help="Seed for the train/val/test split ONLY. Defaults to --seed. "
                         "Vary (e.g. 0..4) to produce repeated stratified splits. "
                         "Output file is tagged: {cancer}_graph_s{split_seed}.pt")
parser.add_argument("--train_ratio", type=float, default=0.7,
                    help="Train split ratio. Default 0.7 (70/10/20 split).")
# FIX #1: val_ratio for 70/10/20 split to eliminate model-selection leakage.
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
# FIX #2: PCA before KNN graph construction (optional).
# PCA removes noise dimensions → cleaner topology. Node features (Data.x)
# remain full-dimensional; PCA only affects WHICH patients are neighbors.
parser.add_argument("--pca_graph",   type=int,   default=None,
                    help="If set, apply PCA to this many components before building KNN "
                         "graph. Cleaner topology, does not change node feature dims. "
                         "Recommended: 64. Default=None (no PCA).")
args = parser.parse_args()

# Split seed defaults to the global seed (backward compatible).
split_seed = args.split_seed if args.split_seed is not None else args.seed

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
# 3. FEATURE SELECTION (MAD-based, fit on train, RAW values)
# ─────────────────────────────────────────────
def select_top_mad(X: np.ndarray, train_idx: np.ndarray, k: int) -> np.ndarray:
    """
    Select k features with highest MAD, computed on TRAIN ONLY, on RAW values.

    MAD (Median Absolute Deviation):
        MAD(X) = median( |Xi - median(X)| )

    IMPORTANT (FIX P1): MAD must be computed BEFORE StandardScaler.
      MAD(z-scored X) = MAD(X)/σ, and for a roughly symmetric feature
      MAD(X) ≈ 0.6745·σ, so post-scaling MAD collapses toward a constant for
      every feature — the dispersion ranking is destroyed and selection
      degenerates into "deviation-from-Gaussianity". Selecting on raw values
      preserves the intended "most variable features" semantics.

    Why MAD instead of variance?
      - Robust against extreme outliers (CNV spikes, expression bursts).
      - All-zero features → MAD=0 → never selected (auto-filters dead miRNA).
      - Partially-zero features → MAD>0 → kept (real silencing signal).
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
      - Isolated nodes (no mutual neighbour) still receive a self-loop from
        GATConv (add_self_loops=True), so they pass their own features through.

    If pca_components is set:
      - PCA is fit on train_idx ONLY (no leakage), transform applied to all.
      - Node features (Data.x) are NOT reduced — only topology is affected.

    NOTE (transductivity): cosine similarity and topk are computed over all
    patients, so test-node features participate in topology. This is the
    standard transductive node-classification setting; disclose it in the
    paper and, ideally, also report an inductive variant where test nodes are
    attached only at inference.
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
    np.fill_diagonal(sim, -1)               # prevent self-loops in constructed edges

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
df_mirna = load_omic(f'{cancer_type}_miRNA.csv')   # NaN → train-median impute downstream
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

# FIX P2 (corrected): robust label alignment.
#
# IMPORTANT: this dataset's *_label.csv is a BARE label vector — a single
# column of class ids with NO patient-barcode column — so it must be aligned
# POSITIONALLY to the omics order (exactly as the original pipeline did).
# A previous revision read it with index_col=0, which consumed the only column
# as the index (leaving 0 data columns) and crashed on .iloc[:, 0]. Do NOT use
# index_col here.
#
# This loader auto-detects whether a real barcode column is present:
#   - If a column's 16-char values overlap the omics patient set -> align by
#     barcode (order-independent, safe).
#   - Otherwise -> positional alignment to the mRNA patient order, with a
#     length-match assertion so a true mismatch fails loudly instead of
#     silently scrambling labels.
labels_df = pd.read_csv(os.path.join(base_path, f'{cancer_type}_label.csv'))

# Wide layout (1 row x many patient columns) -> make it column-shaped.
if labels_df.shape[0] == 1 and labels_df.shape[1] > 1:
    labels_df = labels_df.T.reset_index()

common_set = set(common)

# Detect a patient-barcode column by overlap with the omics patient set.
barcode_col = None
for c in labels_df.columns:
    vals = labels_df[c].astype(str).str[:16]
    if len(common_set & set(vals)) >= 0.5 * len(common_set):
        barcode_col = c
        break

if barcode_col is not None:
    idx_bc     = labels_df[barcode_col].astype(str).str[:16]
    label_cols = [c for c in labels_df.columns if c != barcode_col]
    label_vec  = labels_df[label_cols[-1]] if label_cols else labels_df[barcode_col]
    label_vec  = pd.Series(label_vec.values, index=idx_bc.values)
    missing    = common_set - set(idx_bc.values)
    assert not missing, f"{len(missing)} common patients missing from label file."
    label_series = label_vec.loc[common]
    print("  → Labels aligned by patient barcode.")
else:
    # Bare label vector: positional alignment to mRNA order (original behaviour).
    label_vec = labels_df.iloc[:, -1]
    assert len(label_vec) == len(df_mrna.index), (
        f"#labels ({len(label_vec)}) != #mRNA patients ({len(df_mrna.index)}); "
        "cannot positionally align labels to omics.")
    label_vec = pd.Series(label_vec.values, index=df_mrna.index)
    label_series = label_vec.loc[common]
    print("  ⚠  Label file has no patient barcodes → POSITIONAL alignment to "
          "mRNA column order. Verify label rows match the omics ordering.")

y   = LabelEncoder().fit_transform(label_series.values.ravel())
idx = np.arange(len(common))

# Step 1: carve out test (20%)  — uses split_seed
test_size = 1.0 - args.train_ratio - args.val_ratio
train_val_idx, test_idx = train_test_split(
    idx, test_size=test_size, stratify=y, random_state=split_seed
)

# Step 2: split remainder into train (70%) and val (10%)  — uses split_seed
val_size_adjusted = args.val_ratio / (args.train_ratio + args.val_ratio)
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=val_size_adjusted,
    stratify=y[train_val_idx], random_state=split_seed
)

train_mask = torch.zeros(len(common), dtype=torch.bool)
val_mask   = torch.zeros(len(common), dtype=torch.bool)
test_mask  = torch.zeros(len(common), dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"  → split_seed={split_seed}")
print(f"  → Train: {train_mask.sum().item()}, "
      f"Val: {val_mask.sum().item()}, "
      f"Test: {test_mask.sum().item()}")
print(f"  → Class distribution (full): {np.bincount(y)}")
print(f"  → Class distribution (train): {np.bincount(y[train_idx])}")

for cls_i, cnt in enumerate(np.bincount(y)):
    if cnt < 5:
        print(f"  ⚠  Class {cls_i} has only {cnt} total samples — "
              f"per-class evaluation will be unreliable.")


# ─────────────────────────────────────────────
# 7. PREPROCESS: IMPUTE → MAD-SELECT (RAW) → SCALE
# ─────────────────────────────────────────────
print("[3/5] Preprocessing features...")

def preprocess(df: pd.DataFrame, topk, name: str) -> np.ndarray:
    """
    Order matters (FIX P1): impute → MAD-select on RAW (train) → scale (train).
    Selecting AFTER z-scoring collapses MAD toward a constant and destroys the
    dispersion ranking, so feature selection must precede StandardScaler.
    No double normalization, no leakage:
      0. Train-median imputation for NaN (fit on train only)
      1. MAD feature selection on RAW values, computed on TRAIN only
      2. StandardScaler fit on TRAIN → transform train/val/test
    """
    X = df.loc[common].values.astype(np.float32)

    # Step 0: Impute NaN with train median (ALL omics, consistent, vectorised)
    if np.isnan(X).any():
        train_median = np.nanmedian(X[train_idx], axis=0)
        nan_r, nan_c = np.where(np.isnan(X))
        X[nan_r, nan_c] = np.take(train_median, nan_c)
        print(f"  {name}: NaN → train-median imputation ({len(nan_r)} cells)")

    # Step 1: MAD feature selection on RAW values, train-only
    if topk is not None and topk < X.shape[1]:
        X = select_top_mad(X, train_idx, topk)
        print(f"  {name}: {df.shape[1]} → {X.shape[1]} features (MAD top-{topk}, raw)")
    else:
        print(f"  {name}: {X.shape[1]} features (no reduction)")

    # Step 2: Scale — fit on train only
    scaler = StandardScaler()
    X[train_idx] = scaler.fit_transform(X[train_idx])
    X[val_idx]   = scaler.transform(X[val_idx])
    X[test_idx]  = scaler.transform(X[test_idx])

    return X

X_mrna  = preprocess(df_mrna,  args.topk_mrna,  "mRNA")
X_mirna = preprocess(df_mirna, args.topk_mirna, "miRNA")
X_methy = preprocess(df_methy, args.topk_methy, "Methy")
X_cnv   = preprocess(df_cnv,   args.topk_cnv,   "CNV")


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
        val_mask   = val_mask,
        test_mask  = test_mask,
    )

dataset = {
    "mRNA":  create_data(X_mrna,  "mRNA"),
    "miRNA": create_data(X_mirna, "miRNA"),
    "Methy": create_data(X_methy, "Methy"),
    "CNV":   create_data(X_cnv,   "CNV"),
}


# ─────────────────────────────────────────────
# 9. SAVE  (filename tagged with split_seed)
# ─────────────────────────────────────────────
print("[5/5] Saving...")
save_path = os.path.join(repo_path, f"{cancer_type}_graph_s{split_seed}.pt")
torch.save(dataset, save_path)
print(f"\n✅ DONE! Saved to: {save_path}")
print(f"   num_classes for {cancer_type}: {int(y_tensor.max().item()) + 1}")
print(f"   Split (seed={split_seed}): {train_mask.sum().item()} train / "
      f"{val_mask.sum().item()} val / "
      f"{test_mask.sum().item()} test")
print(f"   In main.py pass --split_seed {split_seed} to load this file.")
