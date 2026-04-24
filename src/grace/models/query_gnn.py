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


def combined_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    lam: float = 0.5,
    neg_ratio: int = 5,
    temperature: float = 0.1,
) -> torch.Tensor:
    return bce_loss(logits, pos_mask, neg_ratio) + lam * infonce_loss(
        logits, pos_mask, temperature
    )
