"""Evaluate saved GNN checkpoints on retrieval-only validation metrics."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from grace.datasets.graph_dataset import GraphMatchDataset, longmem_samples_to_graph_rows  # noqa: E402
from grace.datasets.longmemeval import load_longmemeval_s, split_indices  # noqa: E402
from grace.models.query_gnn import QueryGNN  # noqa: E402


def collate(batch):
    return batch


@torch.no_grad()
def eval_recall(model, loader, device, k: int) -> float:
    model.eval()
    hits, tot = 0, 0
    for batch in loader:
        for x, edge_index, q, pos_mask, _, _, _ in batch:
            x, edge_index = x.to(device), edge_index.to(device)
            q, pos_mask = q.to(device), pos_mask.to(device)
            logits = model(x, edge_index, q)
            topk = torch.topk(logits, k=min(k, int(logits.size(0)))).indices
            hits += int(pos_mask[topk].any().item())
            tot += 1
    return hits / max(tot, 1)


@torch.no_grad()
def eval_strong_hit(model, loader, device, k: int) -> float:
    model.eval()
    hits, tot = 0, 0
    for batch in loader:
        for x, edge_index, q, _, _, strong_pos_mask, _ in batch:
            if not strong_pos_mask.any():
                continue
            x, edge_index = x.to(device), edge_index.to(device)
            q = q.to(device)
            strong_pos_mask = strong_pos_mask.to(device)
            logits = model(x, edge_index, q)
            topk = torch.topk(logits, k=min(k, int(logits.size(0)))).indices
            hits += int(strong_pos_mask[topk].any().item())
            tot += 1
    return hits / max(tot, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/longmemeval/longmemeval_s.json")
    p.add_argument("--checkpoint_glob", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk_graph", type=int, default=5)
    p.add_argument("--graph_session_window", type=int, default=1)
    p.add_argument("--graph_session_semantic_topk", type=int, default=0)
    p.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--gnn_arch", choices=("sage", "sage_res", "sage_skip", "sage_qa", "gat"), default="sage")
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_longmemeval_s(args.data)
    _, val_idx = split_indices(len(samples), 0.8, args.seed)
    val_s = [samples[i] for i in val_idx]
    val_rows = longmem_samples_to_graph_rows(
        val_s,
        topk=args.topk_graph,
        sbert=args.sbert,
        weak_positive_weight=0.0,
        session_window=args.graph_session_window,
        session_semantic_topk=args.graph_session_semantic_topk,
    )
    val_ds = GraphMatchDataset(val_rows, sbert=None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    paths = sorted(glob.glob(args.checkpoint_glob))
    rows = []
    for ckpt in paths:
        model = QueryGNN(
            in_dim=384,
            hidden_dim=args.hidden,
            num_layers=args.num_layers,
            query_dim=384,
            arch=args.gnn_arch,
            gat_heads=args.gat_heads,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        r = eval_recall(model, val_loader, device, k=args.k)
        strong = eval_strong_hit(model, val_loader, device, k=args.k)
        row = {"checkpoint": ckpt, f"val_R@{args.k}": r, f"val_StrongHit@{args.k}": strong}
        rows.append(row)
        print(
            f"{ckpt}\tval_R@{args.k}={r:.3f}\t"
            f"val_StrongHit@{args.k}={strong:.3f}",
            flush=True,
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] saved {out}")


if __name__ == "__main__":
    main()
