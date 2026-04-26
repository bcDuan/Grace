"""Query-conditioned GraphSAGE retriever for GRACE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class QueryConditionedSAGEConv(MessagePassing):
    """One GraphSAGE layer conditioned on a graph-level query vector."""

    def __init__(
        self, in_dim: int, out_dim: int, query_dim: int, dropout: float = 0.2
    ):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_dim * 2 + query_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        agg = self.propagate(edge_index, x=x)
        q_expand = q.unsqueeze(0).expand(x.size(0), -1)
        h = torch.cat([x, agg, q_expand], dim=-1)
        h = self.lin(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class QueryGNN(nn.Module):
    """Stacked query-conditioned GraphSAGE + MLP scoring head."""

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 2,
        query_dim: int = 384,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            QueryConditionedSAGEConv(in_dim, hidden_dim, query_dim, dropout)
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                QueryConditionedSAGEConv(
                    hidden_dim, hidden_dim, query_dim, dropout
                )
            )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + query_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, q)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index, q)
        q_expand = q.unsqueeze(0).expand(h.size(0), -1)
        logits = self.score_head(torch.cat([h, q_expand], dim=-1)).squeeze(-1)
        return logits


def bce_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_ratio: int = 5,
) -> torch.Tensor:
    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
    neg_pool = (~pos_mask).nonzero(as_tuple=True)[0]
    if len(pos_idx) == 0 or len(neg_pool) == 0:
        return logits.new_zeros(())
    n_neg = min(len(pos_idx) * neg_ratio, len(neg_pool))
    neg_idx = neg_pool[
        torch.randperm(len(neg_pool), device=logits.device)[:n_neg]
    ]
    pos_logits = logits[pos_idx]
    neg_logits = logits[neg_idx]
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits)
    )
    return pos_loss + neg_loss


def weighted_bce_loss(
    logits: torch.Tensor,
    label_weights: torch.Tensor,
    neg_ratio: int = 5,
) -> torch.Tensor:
    """Weighted BCE for answer-aware weak supervision.

    Nodes with label_weights > 0 are positive nodes. Weight magnitude controls
    how strongly a positive should be learned, e.g. answer-bearing turns get
    1.0 while same-session context turns may get 0.2.
    """
    pos_idx = (label_weights > 0).nonzero(as_tuple=True)[0]
    neg_pool = (label_weights <= 0).nonzero(as_tuple=True)[0]
    if len(pos_idx) == 0 or len(neg_pool) == 0:
        return logits.new_zeros(())
    n_neg = min(len(pos_idx) * neg_ratio, len(neg_pool))
    neg_idx = neg_pool[
        torch.randperm(len(neg_pool), device=logits.device)[:n_neg]
    ]

    pos_logits = logits[pos_idx]
    pos_targets = torch.ones_like(pos_logits)
    pos_weights = label_weights[pos_idx].to(logits.dtype)
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits,
        pos_targets,
        weight=pos_weights,
        reduction="sum",
    ) / pos_weights.sum().clamp_min(1.0)

    neg_logits = logits[neg_idx]
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits,
        torch.zeros_like(neg_logits),
    )
    return pos_loss + neg_loss


def infonce_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
    if len(pos_idx) == 0:
        return logits.new_zeros(())
    scaled = logits / temperature
    log_denom = torch.logsumexp(scaled, dim=0)
    log_num = scaled[pos_idx]
    return -(log_num - log_denom).mean()


def answer_pairwise_ranking_loss(
    logits: torch.Tensor,
    label_weights: torch.Tensor,
    margin: float = 0.5,
    hard_negatives: int = 16,
) -> torch.Tensor:
    """Rank answer-bearing turns above weaker context and hard negatives.

    This loss is only active when answer-aware labels found strong positives
    (label weight >= 1.0). It directly targets the QA failure mode where the
    model retrieves the right session but ranks a non-answer turn above the
    answer-bearing turn.
    """
    strong_idx = (label_weights >= 1.0).nonzero(as_tuple=True)[0]
    cand_idx = (label_weights < 1.0).nonzero(as_tuple=True)[0]
    if len(strong_idx) == 0 or len(cand_idx) == 0:
        return logits.new_zeros(())

    if hard_negatives > 0 and len(cand_idx) > hard_negatives:
        cand_scores = logits[cand_idx].detach()
        top_local = torch.topk(cand_scores, k=hard_negatives).indices
        cand_idx = cand_idx[top_local]

    pos_logits = logits[strong_idx].unsqueeze(1)
    neg_logits = logits[cand_idx].unsqueeze(0)
    return F.relu(margin - pos_logits + neg_logits).mean()


def combined_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    label_weights: torch.Tensor | None = None,
    lam: float = 0.5,
    neg_ratio: int = 5,
    temperature: float = 0.1,
    rank_lam: float = 0.0,
    rank_margin: float = 0.5,
    hard_negatives: int = 16,
) -> torch.Tensor:
    if label_weights is None:
        bce = bce_loss(logits, pos_mask, neg_ratio)
        contrast_mask = pos_mask
        rank = logits.new_zeros(())
    else:
        bce = weighted_bce_loss(logits, label_weights, neg_ratio)
        contrast_mask = label_weights >= 1.0
        if not contrast_mask.any():
            contrast_mask = pos_mask
        rank = answer_pairwise_ranking_loss(
            logits,
            label_weights,
            margin=rank_margin,
            hard_negatives=hard_negatives,
        )
    return bce + lam * infonce_loss(logits, contrast_mask, temperature) + rank_lam * rank
