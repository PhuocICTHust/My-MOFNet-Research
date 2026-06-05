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
    """

    def __init__(self, in_dim: int, hid: int, dropout: float = 0.3):
        super().__init__()

        self.conv1 = GATConv(in_dim, hid, heads=2, concat=False)
        self.bn1 = nn.BatchNorm1d(hid)

        self.conv2 = GATConv(hid, hid, heads=2, concat=False)
        self.bn2 = nn.BatchNorm1d(hid)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2)

        x = F.dropout(x, p=self.dropout, training=self.training)

        z1 = self.conv1(x, edge_index)
        z1 = self.bn1(z1)
        z1 = F.elu(z1)

        z1 = F.dropout(z1, p=self.dropout, training=self.training)

        z2 = self.conv2(z1, edge_index)
        z2 = self.bn2(z2)

        z2 = F.elu(z2 + z1)

        return z2


# ─────────────────────────────────────────────
# 2. ATTENTION FUSION
# ─────────────────────────────────────────────
class AttentionFusion(nn.Module):

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

        z = torch.stack(z_list, dim=1)

        w = F.softmax(
            self.score(z) / self.temperature,
            dim=1,
            )

        entropy = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()

        fused = (z * w).sum(dim=1)

        return fused, w.squeeze(-1), entropy


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

        self.enc_mrna = LightGATEncoder(
            data["mRNA"].num_node_features,
            hid,
            drop,
        )

        self.enc_mirna = LightGATEncoder(
            data["miRNA"].num_node_features,
            hid,
            drop,
        )

        self.enc_methy = LightGATEncoder(
            data["Methy"].num_node_features,
            hid,
            drop,
        )

        self.enc_cnv = LightGATEncoder(
            data["CNV"].num_node_features,
            hid,
            drop,
        )

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

        logits = self.classifier(fused)

        return logits, entropy

    def get_attention_weights(self):

        return self.last_attn