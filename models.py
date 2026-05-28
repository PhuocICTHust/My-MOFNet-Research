"""
models.py — MOFNet-Pro (Stable, Research-Grade)

Architecture:
  LightGATEncoder × 4  →  AttentionFusion  →  Linear classifier
  + FocalLoss (class imbalance)

Fixes vs previous version:
  1. AttentionFusion thêm entropy regularization → chống modality collapse
     forward() trả về (fused, weights, entropy) để main.py dùng trong loss
  2. FocalLoss giữ nguyên, class weights được clamp ở main.py

Data context (BRCA):
  N=671, 5 classes [353, 42, 132, 31, 113], 4 omics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge
from typing import Optional

# ─────────────────────────────────────────────
# 1. LIGHT GAT ENCODER
# ─────────────────────────────────────────────
class LightGATEncoder(nn.Module):
    """
    1-layer GAT — tránh over-smoothing trên N=671.
    heads=2, concat=False → output dim = hid (không phình).
    DropEdge p=0.2 chỉ lúc training.
    ELU thay ReLU: mượt hơn với output âm từ BatchNorm.
    """
    def __init__(self, in_dim: int, hid: int, dropout: float = 0.3):
        super().__init__()
        self.conv    = GATConv(in_dim, hid, heads=2, concat=False)
        self.bn      = nn.BatchNorm1d(hid)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x, edge_index)
        x = self.bn(x)
        return F.elu(x)


# ─────────────────────────────────────────────
# 2. ATTENTION FUSION (entropy-regularized)
# ─────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Attention fusion với entropy regularization.

    Tại sao entropy reg?
      Trên TCGA, mRNA thường dominate vì feature count cao nhất.
      Entropy penalty khuyến khích model phân bổ attention đều hơn,
      tránh collapse về 1 omics.

    forward() trả về:
      fused   [N, D]  — đặc trưng sau fusion
      weights [N, M]  — attention weights (dùng cho explainability)
      entropy scalar  — penalty term cho loss (× 0.01)
    """
    def __init__(self, dim: int, n_modality: int = 4):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, z_list: list):
        z = torch.stack(z_list, dim=1)          # [N, M, D]
        w = F.softmax(self.score(z), dim=1)      # [N, M, 1]

        # Entropy reg: max entropy = log(M) ≈ 1.39 với M=4
        entropy = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()

        fused = (z * w).sum(dim=1)               # [N, D]
        return fused, w.squeeze(-1), entropy     # [N,D], [N,M], scalar


# ─────────────────────────────────────────────
# 3. FOCAL LOSS
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss với class weights.
    gamma=2.0: phạt nặng các ca khó (minority class 1=42, class 3=31).
    alpha = class_weights đã clamp từ main.py.
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
    MOFNet-Pro: 4 × LightGATEncoder → AttentionFusion → Linear classifier.

    forward() trả về (logits, entropy) để main.py tính:
        loss = FocalLoss(logits, y) + 0.01 * entropy
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

        # Lưu lại để dùng cho explainability sau training
        self.last_attn = attn_weights.detach()

        return self.classifier(fused), entropy

    def get_attention_weights(self) -> torch.Tensor:
        """Trả về [N, 4] attention weights sau forward — dùng để plot omics importance."""
        return self.last_attn
