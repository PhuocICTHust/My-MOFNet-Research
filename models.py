import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import dropout_edge
from typing import Optional


# ─────────────────────────────────────────────
# 1. LIGHT GNN ENCODER (2 layers + residual skip)  — GAT or GCN
# ─────────────────────────────────────────────
class LightGATEncoder(nn.Module):
    """
    2-layer graph encoder with residual skip connection.

    conv_type ("gat" | "gcn"):  PATCH 1 — lets you run the GAT-vs-GCN ablation.
      - "gat": GATConv, heads=2, concat=False (learns edge attention; can prune
               noisy KNN edges; ~2x the parameters).
      - "gcn": GCNConv (fixed symmetric-normalised propagation; ~50% fewer
               parameters → stronger regulariser on the smallest cohorts
               GBM/OV/COAD, N≈244–284). The class name is kept for backward
               compatibility with existing imports/checkpoints.

    LayerNorm (not BatchNorm1d): the model runs over the FULL graph
    (train+val+test). BatchNorm in train() mode computes batch stats over every
    node and updates running stats from them → train-node activations (and the
    train gradient) depend on val/test feature magnitudes, and eval-time running
    stats are polluted by test nodes. That is avoidable test-statistic leakage.
    LayerNorm normalises per node over the feature dim: no cross-node stats, no
    running stats, no leakage — and is more stable in full-batch transductive GNNs.
    """

    def __init__(self, in_dim: int, hid: int, dropout: float = 0.3,
                 conv_type: str = "gat"):
        super().__init__()

        def make_conv(cin, cout):
            if conv_type == "gcn":
                return GCNConv(cin, cout)
            return GATConv(cin, cout, heads=2, concat=False)

        self.conv_type = conv_type
        self.conv1 = make_conv(in_dim, hid)
        self.norm1 = nn.LayerNorm(hid)

        self.conv2 = make_conv(hid, hid)
        self.norm2 = nn.LayerNorm(hid)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2)

        x = F.dropout(x, p=self.dropout, training=self.training)

        z1 = self.conv1(x, edge_index)
        z1 = self.norm1(z1)
        z1 = F.elu(z1)

        z1 = F.dropout(z1, p=self.dropout, training=self.training)

        z2 = self.conv2(z1, edge_index)
        z2 = self.norm2(z2)

        z2 = F.elu(z2 + z1)

        return z2


# ─────────────────────────────────────────────
# 2. ATTENTION FUSION
# ─────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Per-patient soft gating over the 4 modality representations.

    NOTE: this is a single bias-free linear gate (Linear(dim, 1)) with a
    softmax over modalities, regularised toward uniformity by the entropy term
    in main.py. In practice it behaves close to a learned-tilt mean-pool — a
    stable fusion baseline, but lighter than full query/key cross-attention.
    """

    def __init__(
            self,
            dim: int,
            n_modality: int = 4,
            temperature: float = 1.0,
    ):
        super().__init__()

        self.score = nn.Linear(dim, 1, bias=False)
        self.temperature = temperature

    def forward(self, z_list):

        z = torch.stack(z_list, dim=1)            # [N, M, dim]

        w = F.softmax(
            self.score(z) / self.temperature,
            dim=1,
            )                                     # [N, M, 1]

        entropy = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()

        fused = (z * w).sum(dim=1)                # [N, dim]

        return fused, w.squeeze(-1), entropy


# ─────────────────────────────────────────────
# 2b. CROSS-OMICS FUSION (VCDN-lite)  — PATCH 2 (optional)
# ─────────────────────────────────────────────
class CrossOmicsFusion(nn.Module):
    """
    Lightweight VCDN-style fusion in LABEL space (replaces AttentionFusion +
    classifier when --fusion cross).

    Idea (from the paper's VCDN): instead of a soft weighted-sum of modality
    *features*, model the correlation between per-omic *class distributions*.
    Each omic gets its own linear head → per-omic logits → softmax probs. We
    then form the pairwise outer products p_i ⊗ p_j (the cross-omics
    co-occurrence the paper calls "view correlation") and learn a small linear
    map over them. Final logits = mean(per-omic logits) + cross-correlation term.

    Why this is the right lever (not GAT-vs-GCN): on BRCA the paper's full model
    (with VCDN) beats its attention-fusion ablation by ~4 macro-F1 points, while
    GAT vs GCN is roughly tied. The gap lives in fusion.

    Kept lightweight & stable for Small-N:
      - pairwise (not full C^M tensor): 6 pairs × C² inputs for 4 omics.
      - e.g. C=5 → 6·25 = 150 → C  ⇒ ~150 extra params (vs C^4 = 625 for full VCDN).
    CAUTION: the extra interaction terms can overfit the tiniest cohorts
    (COAD class-of-4, GBM). Validate per-cohort with the 5×5 protocol before
    trusting it; if it underperforms attn there, keep --fusion attn for those.
    """

    def __init__(self, dim: int, n_modality: int = 4, n_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        self.M = n_modality
        self.C = n_classes
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
             for _ in range(n_modality)]
        )
        n_pairs = n_modality * (n_modality - 1) // 2
        self.cross = nn.Linear(n_pairs * n_classes * n_classes, n_classes)

    def forward(self, z_list):
        logits = [head(z) for head, z in zip(self.heads, z_list)]   # M × [N, C]
        probs  = [F.softmax(l, dim=1) for l in logits]

        pair = []
        for i in range(self.M):
            for j in range(i + 1, self.M):
                outer = torch.bmm(probs[i].unsqueeze(2),
                                  probs[j].unsqueeze(1))            # [N, C, C]
                pair.append(outer.reshape(outer.size(0), -1))

        cross_logits = self.cross(torch.cat(pair, dim=1))          # [N, C]
        mean_logits  = torch.stack(logits, dim=0).mean(dim=0)      # [N, C]
        final        = mean_logits + cross_logits

        # Per-omic mean confidence as an attention proxy (keeps evaluate()/logging happy)
        attn = torch.stack(probs, dim=1).mean(dim=-1)              # [N, M]
        entropy = torch.tensor(0.0, device=final.device)
        return final, attn, entropy


# ─────────────────────────────────────────────
# 3. FOCAL LOSS
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):

    def __init__(
            self,
            gamma: float = 2.0,
            alpha: Optional[torch.Tensor] = None,
            label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
    ):

        n_classes = logits.size(1)

        if self.label_smoothing > 0:

            smooth = self.label_smoothing / max(
                n_classes - 1,
                1,
                )

            soft = torch.full_like(logits, smooth)

            soft.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.label_smoothing,
                )

            log_p = F.log_softmax(logits, dim=1)

            if self.alpha is not None:

                w = self.alpha[targets]

                ce = -(soft * log_p).sum(dim=1) * w

            else:

                ce = -(soft * log_p).sum(dim=1)

        else:

            ce = F.cross_entropy(
                logits,
                targets,
                weight=self.alpha,
                reduction="none",
            )

        with torch.no_grad():

            pt = torch.exp(
                -F.cross_entropy(
                    logits,
                    targets,
                    reduction="none",
                )
            )

        loss = ((1 - pt) ** self.gamma) * ce

        return loss.mean()


# ─────────────────────────────────────────────
# 4. PANCANCER MODEL
# ─────────────────────────────────────────────
class PanCancerModel(nn.Module):

    def __init__(self, args, data):
        super().__init__()

        hid = args.nhid
        drop = args.dropout_ratio
        conv_type = getattr(args, "conv_type", "gat")      # PATCH 1
        self.fusion_mode = getattr(args, "fusion", "attn")  # PATCH 2

        self.enc_mrna = LightGATEncoder(
            data["mRNA"].num_node_features,
            hid,
            drop,
            conv_type,
        )

        self.enc_mirna = LightGATEncoder(
            data["miRNA"].num_node_features,
            hid,
            drop,
            conv_type,
        )

        self.enc_methy = LightGATEncoder(
            data["Methy"].num_node_features,
            hid,
            drop,
            conv_type,
        )

        self.enc_cnv = LightGATEncoder(
            data["CNV"].num_node_features,
            hid,
            drop,
            conv_type,
        )

        if self.fusion_mode == "cross":
            # VCDN-lite: fuses in label space and produces logits directly.
            self.fusion = CrossOmicsFusion(
                hid,
                n_modality=4,
                n_classes=args.num_classes,
                dropout=drop,
            )
            self.classifier = None
        else:
            temp = getattr(args, "attention_temp", 1.0)
            self.fusion = AttentionFusion(
                hid,
                n_modality=4,
                temperature=temp,
            )
            self.classifier = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(
                    hid,
                    args.num_classes,
                ),
            )

    def forward(self, d1, d2, d3, d4):

        z1 = self.enc_mrna(
            d1.x,
            d1.edge_index,
        )

        z2 = self.enc_mirna(
            d2.x,
            d2.edge_index,
        )

        z3 = self.enc_methy(
            d3.x,
            d3.edge_index,
        )

        z4 = self.enc_cnv(
            d4.x,
            d4.edge_index,
        )

        fused, attn_weights, entropy = self.fusion(
            [z1, z2, z3, z4]
        )

        self.last_attn = attn_weights.detach()

        if self.fusion_mode == "cross":
            # CrossOmicsFusion already returns final logits in `fused`.
            logits = fused
        else:
            logits = self.classifier(fused)

        return logits, entropy

    def get_attention_weights(self):

        return self.last_attn
