import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge
from typing import Optional


# ─────────────────────────────────────────────
# 1. LIGHT GAT ENCODER (2 layers + residual skip)
# ─────────────────────────────────────────────
class LightGATEncoder(nn.Module):
    """
    2-layer GAT with residual skip connection.

    Architecture:
        Layer 1: in_dim → hid  (heads=2, concat=False)
        Layer 2: hid    → hid  (heads=2, concat=False)
        Skip:    z2 = ELU( BN(GAT2(z1)) + z1 )

    Why 2 layers vs 1?
      - 1-layer: only sees 1-hop neighbors (direct edges).
      - 2-layer: sees 2-hop neighborhoods — captures indirect relations between
        patients connected through a common "neighbor" patient in the graph.
        On TCGA (N=244–671), 2 hops is enough to capture local community structure
        without reaching the full graph (over-smoothing).

    Why NOT 3 layers?
      - On small graphs (N≤671), 3-hop neighborhoods often cover the entire graph
        → all node embeddings converge → over-smoothing → accuracy collapse.

    Why residual skip?
      - z2 = BN(GAT2(z1)) + z1: if GAT2 learns nothing useful, residual
        passes z1 through unchanged → training is always stable.
      - Prevents degradation when adding the second layer.

    heads=2, concat=False: output dim = hid (not 2×hid), keeping memory constant.
    DropEdge p=0.2: randomly remove edges during training → acts as graph-level dropout,
    improves robustness on small TCGA graphs.
    ELU: smoother than ReLU for negative values produced by BatchNorm.
    """
    def __init__(self, in_dim: int, hid: int, dropout: float = 0.3):
        super().__init__()
        # Layer 1
        self.conv1   = GATConv(in_dim, hid, heads=2, concat=False)
        self.bn1     = nn.BatchNorm1d(hid)
        # Layer 2
        self.conv2   = GATConv(hid, hid, heads=2, concat=False)
        self.bn2     = nn.BatchNorm1d(hid)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2)

        # ── Layer 1 ─────────────────────────────────────────────────────
        x  = F.dropout(x, p=self.dropout, training=self.training)
        z1 = F.elu(self.bn1(self.conv1(x, edge_index)))

        # ── Layer 2 + residual skip ─────────────────────────────────────
        z1 = F.dropout(z1, p=self.dropout, training=self.training)
        z2 = self.bn2(self.conv2(z1, edge_index))
        z2 = F.elu(z2 + z1)       # residual: skip z1 around layer 2

        return z2


# ─────────────────────────────────────────────
# 2. ATTENTION FUSION (entropy-regularized)
# ─────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Soft attention fusion with entropy regularization.

    Why entropy regularization?
      On TCGA, mRNA tends to dominate attention because it has the highest
      feature count after selection. Without regularization, the model collapses
      to a single-omics classifier (ignoring miRNA, Methy, CNV).
      Entropy penalty = -(w * log(w)).sum() encourages w to be more uniform
      across the 4 modalities, forcing the model to leverage all omics.
      Max entropy for M=4 modalities = log(4) ≈ 1.39.

    Note: entropy is MAXIMIZED (subtracted from loss in main.py):
        loss = focal_loss - entropy_weight * entropy

    forward() returns:
      fused   [N, D]  — fused patient embeddings
      weights [N, M]  — per-patient attention weights (for explainability)
      entropy scalar  — penalty term passed to main.py
    """
    def __init__(self, dim: int, n_modality: int = 4):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, z_list: list):
        z = torch.stack(z_list, dim=1)          # [N, M, D]
        w = F.softmax(self.score(z), dim=1)     # [N, M, 1]  (softmax over M)

        entropy = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()   # scalar

        fused = (z * w).sum(dim=1)              # [N, D]
        return fused, w.squeeze(-1), entropy    # [N,D], [N,M], scalar


# ─────────────────────────────────────────────
# 3. FOCAL LOSS
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss with class weights.

    Standard cross-entropy treats all samples equally.
    Focal Loss down-weights easy samples (high confidence) and focuses
    training on hard/misclassified samples (minority classes).

    gamma=2.0: (1 - pt)^gamma acts as a modulating factor.
      - Easy sample (pt=0.9): factor = (0.1)^2 = 0.01 → almost ignored.
      - Hard sample  (pt=0.1): factor = (0.9)^2 = 0.81 → strongly penalized.

    alpha = class_weights (balanced, clamped [0.5, 5.0] in main.py):
      Combined with gamma, gives extra weight to minority classes that are
      also hard to classify (double benefit for rare cancer subtypes).
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────
# 4. PANCANCER MODEL
# ─────────────────────────────────────────────
class PanCancerModel(nn.Module):
    """
    4 × LightGATEncoder → AttentionFusion → Linear classifier.

    Each encoder independently processes one omic modality on its own
    patient similarity graph (built from that omic's feature space).
    AttentionFusion learns to weight the 4 embeddings per patient.

    forward() returns (logits, entropy) so main.py computes:
        loss = FocalLoss(logits, y) - entropy_weight * entropy
    """
    def __init__(self, args, data: dict):
        super().__init__()
        hid  = args.nhid
        drop = args.dropout_ratio

        self.enc_mrna  = LightGATEncoder(data['mRNA'].num_node_features,  hid, drop)
        self.enc_mirna = LightGATEncoder(data['miRNA'].num_node_features, hid, drop)
        self.enc_methy = LightGATEncoder(data['Methy'].num_node_features, hid, drop)
        self.enc_cnv   = LightGATEncoder(data['CNV'].num_node_features,   hid, drop)

        self.fusion = AttentionFusion(hid, n_modality=4)

        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(hid, args.num_classes)
        )

    def forward(self, d1, d2, d3, d4):
        z1 = self.enc_mrna (d1.x, d1.edge_index)
        z2 = self.enc_mirna(d2.x, d2.edge_index)
        z3 = self.enc_methy(d3.x, d3.edge_index)
        z4 = self.enc_cnv  (d4.x, d4.edge_index)

        fused, attn_weights, entropy = self.fusion([z1, z2, z3, z4])

        # Store for post-training explainability
        self.last_attn = attn_weights.detach()

        return self.classifier(fused), entropy

    def get_attention_weights(self) -> torch.Tensor:
        """Returns [N, 4] attention weights — use to plot per-omics importance."""
        return self.last_attn
