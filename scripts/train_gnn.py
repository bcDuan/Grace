"""Train the query-conditioned GNN retriever on LongMemEval evidence labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from grace.datasets.graph_dataset import (  # noqa: E402
    GraphMatchDataset,
    longmem_samples_to_graph_rows,
)
from grace.datasets.longmemeval import load_longmemeval_s, split_indices  # noqa: E402
from grace.qa.judge import LLMJudge  # noqa: E402
from grace.qa.reader import QwenReader  # noqa: E402
from grace.models.query_gnn import QueryGNN, combined_loss  # noqa: E402
from grace.graphs.build import build_sentence_graph  # noqa: E402
from grace.utils.seed import set_seed  # noqa: E402


def collate(batch):
    return batch


def unpack_graph_item(item: tuple):
    """Support both legacy 5-tuples and answer-aware 7-tuples."""
    if len(item) == 5:
        x, edge_index, q, pos_mask, node_texts = item
        label_weights = pos_mask.float()
        strong_pos_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
        return x, edge_index, q, pos_mask, label_weights, strong_pos_mask, node_texts
    return item


def train_one_epoch(
    model,
    loader,
    opt,
    device,
    lam: float,
    neg_ratio: int,
    rank_lam: float,
    rank_margin: float,
    hard_negatives: int,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        opt.zero_grad()
        loss = 0.0
        m = 0
        for item in batch:
            x, edge_index, q, pos_mask, label_weights, _, _ = unpack_graph_item(item)
            x, edge_index = x.to(device), edge_index.to(device)
            q = q.to(device)
            pos_mask = pos_mask.to(device)
            label_weights = label_weights.to(device)
            logits = model(x, edge_index, q)
            loss = loss + combined_loss(
                logits,
                pos_mask,
                label_weights=label_weights,
                lam=lam,
                neg_ratio=neg_ratio,
                rank_lam=rank_lam,
                rank_margin=rank_margin,
                hard_negatives=hard_negatives,
            )
            m += 1
        if m == 0:
            continue
        loss = loss / m
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * m
        n += m
    return total / max(n, 1)


@torch.no_grad()
def eval_recall(model, loader, device, k: int = 5) -> float:
    model.eval()
    hits, tot = 0, 0
    for batch in loader:
        for item in batch:
            x, edge_index, q, pos_mask, _, _, _ = unpack_graph_item(item)
            x, edge_index = x.to(device), edge_index.to(device)
            q, pos_mask = q.to(device), pos_mask.to(device)
            logits = model(x, edge_index, q)
            kk = min(k, int(logits.size(0)))
            topk = torch.topk(logits, k=kk).indices
            hit = pos_mask[topk].any().item()
            hits += int(hit)
            tot += 1
    return hits / max(tot, 1)


@torch.no_grad()
def eval_strong_hit(model, loader, device, k: int = 5) -> float:
    model.eval()
    hits, tot = 0, 0
    for batch in loader:
        for item in batch:
            x, edge_index, q, _, _, strong_pos_mask, _ = unpack_graph_item(item)
            if not strong_pos_mask.any():
                continue
            x, edge_index = x.to(device), edge_index.to(device)
            q = q.to(device)
            strong_pos_mask = strong_pos_mask.to(device)
            logits = model(x, edge_index, q)
            kk = min(k, int(logits.size(0)))
            topk = torch.topk(logits, k=kk).indices
            hits += int(strong_pos_mask[topk].any().item())
            tot += 1
    return hits / max(tot, 1)


def stratified_sample(samples: list, limit: int, seed: int = 42) -> list:
    by_type: dict[str, list] = {}
    for s in samples:
        by_type.setdefault(s.question_type, []).append(s)
    rng = np.random.default_rng(seed)
    for rows in by_type.values():
        rng.shuffle(rows)
    types = sorted(by_type.keys())
    if not types:
        return []
    per_type = max(1, limit // len(types))
    picked = []
    leftovers = []
    for t in types:
        items = by_type[t]
        picked.extend(items[:per_type])
        leftovers.extend(items[per_type:])
    if len(picked) < limit and leftovers:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: limit - len(picked)])
    rng.shuffle(picked)
    return picked[:limit]


def _turn_content(turn: dict) -> str:
    for key in ("content", "text", "value", "message", "body"):
        v = turn.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return str(turn)


def _build_turn_meta(sample) -> dict[int, dict]:
    sessions = sample.raw.get("haystack_sessions") or []
    session_ids = sample.raw.get("haystack_session_ids") or []
    session_dates = sample.raw.get("haystack_dates") or []
    out: dict[int, dict] = {}
    gid = 0
    for si, sess in enumerate(sessions):
        msgs = sess if isinstance(sess, list) else []
        sid = str(session_ids[si]) if si < len(session_ids) else str(si)
        sdate = str(session_dates[si]) if si < len(session_dates) else ""
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            content = _turn_content(msg)
            if not content.strip():
                continue
            out[gid] = {
                "session_id": sid,
                "session_date": sdate,
                "role": str(msg.get("role", "unknown")),
                "content": content,
            }
            gid += 1
    return out


@torch.no_grad()
def eval_qa_accuracy(
    model: QueryGNN,
    val_graphs: list[dict],
    sbert: SentenceTransformer,
    reader: QwenReader,
    judge: LLMJudge,
    device: str,
    k: int = 5,
    reader_batch_size: int = 2,
) -> tuple[float, float]:
    model.eval()
    contexts: list[list[dict]] = []
    questions: list[str] = []
    retrieved_ids: list[list[int]] = []
    session_recalls: list[float] = []
    for item in val_graphs:
        qv = sbert.encode([item["question"]], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        q = torch.tensor(qv, device=device, dtype=torch.float32)
        x = item["x"].to(device)
        ei = item["edge_index"].to(device)
        logits = model(x, ei, q)
        kk = min(k, int(logits.size(0)))
        topk = torch.topk(logits, k=kk).indices.tolist()
        retrieved_ids.append(topk)
        tm = item["turn_meta"]
        ctx = [
            tm.get(
                i,
                {
                    "session_id": str(item["sample"].turn_to_session.get(i, "unknown")),
                    "session_date": "",
                    "role": "unknown",
                    "content": item["sample"].turns[i].text if 0 <= i < len(item["sample"].turns) else "",
                },
            )
            for i in topk
        ]
        contexts.append(ctx)
        questions.append(item["question"])
        gold_sess = set(item["sample"].gold_sessions)
        pred_sess = {item["sample"].turn_to_session.get(i) for i in topk if i in item["sample"].turn_to_session}
        session_recalls.append(len(gold_sess & pred_sess) / max(len(gold_sess), 1))
    preds: list[str] = []
    bs = max(1, reader_batch_size)
    for i in range(0, len(questions), bs):
        preds.extend(reader.answer_batch(questions[i : i + bs], contexts[i : i + bs]))
    judge_samples = [
        {
            "question": item["question"],
            "gold": item["sample"].answer or "",
            "predicted": pred,
            "question_type": item["sample"].question_type,
        }
        for item, pred in zip(val_graphs, preds)
    ]
    import asyncio

    judge_out = asyncio.run(judge.judge_batch(judge_samples, concurrency=10))
    acc = sum(1.0 for j in judge_out if bool(j.get("correct", False))) / max(len(judge_out), 1)
    sess_r5 = sum(session_recalls) / max(len(session_recalls), 1)
    return acc, sess_r5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to longmemeval_s.json")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--neg_ratio", type=int, default=4, help="Negative sampling ratio for BCE.")
    p.add_argument(
        "--rank_lam",
        type=float,
        default=0.0,
        help="Weight for answer-aware pairwise ranking loss.",
    )
    p.add_argument(
        "--rank_margin",
        type=float,
        default=0.5,
        help="Margin for answer-aware pairwise ranking loss.",
    )
    p.add_argument(
        "--hard_negatives",
        type=int,
        default=16,
        help="Number of high-scoring non-answer candidates used by pairwise ranking.",
    )
    p.add_argument(
        "--weak_positive_weight",
        type=float,
        default=0.2,
        help="BCE weight for non-answer turns inside a gold evidence session.",
    )
    p.add_argument("--out", default="experiments/checkpoints/gnn.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk_graph", type=int, default=5)
    p.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--val_limit", type=int, default=30)
    p.add_argument(
        "--val_graph_limit",
        type=int,
        default=0,
        help="Optional cap on validation graphs for retrieval-only smoke runs; 0 means full val split.",
    )
    p.add_argument(
        "--train_limit",
        type=int,
        default=0,
        help="Optional cap on training samples for smoke runs; 0 means full train split.",
    )
    p.add_argument("--val_seed", type=int, default=42)
    p.add_argument(
        "--checkpoint_pattern",
        default="experiments/checkpoints/gnn_epoch{ep}.pt",
        help="Checkpoint path pattern saved at every epoch.",
    )
    p.add_argument("--reader_backend", choices=("vllm", "transformers"), default="transformers")
    p.add_argument("--reader_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--reader_batch_size", type=int, default=2)
    p.add_argument("--judge_backend", choices=("siliconflow", "deepseek", "local_vllm"), default="siliconflow")
    p.add_argument("--judge_cache_dir", default="data/processed/judge_cache")
    p.add_argument(
        "--skip_qa_eval",
        action="store_true",
        help="Skip reader/judge validation and report retrieval-only training logs.",
    )
    p.add_argument("--resume_checkpoint", default="", help="Optional: resume model weights from this checkpoint")
    p.add_argument(
        "--curve_out",
        default="experiments/results/gnn_training_curve.json",
        help="JSON output path for per-epoch curve",
    )
    args = p.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_samples = load_longmemeval_s(args.data)
    n = len(all_samples)
    tr_idx, va_idx = split_indices(n, 0.8, args.seed)
    train_s = [all_samples[i] for i in tr_idx]
    val_s = [all_samples[i] for i in va_idx]
    if args.train_limit and args.train_limit > 0:
        train_s = stratified_sample(train_s, args.train_limit, seed=args.seed)
    val_s_for_rows = val_s
    if args.val_graph_limit and args.val_graph_limit > 0:
        val_s_for_rows = stratified_sample(val_s, args.val_graph_limit, seed=args.val_seed)
    train_rows = longmem_samples_to_graph_rows(
        train_s,
        topk=args.topk_graph,
        sbert=args.sbert,
        weak_positive_weight=args.weak_positive_weight,
    )
    val_rows = longmem_samples_to_graph_rows(
        val_s_for_rows,
        topk=args.topk_graph,
        sbert=args.sbert,
        weak_positive_weight=args.weak_positive_weight,
    )
    if not train_rows:
        print("No training rows (missing evidence or empty turns).")
        return
    train_strong = sum(
        1 for r in train_rows if r.strong_pos_mask is not None and r.strong_pos_mask.any()
    )
    val_strong = sum(
        1 for r in val_rows if r.strong_pos_mask is not None and r.strong_pos_mask.any()
    )
    print(
        f"[label] train_rows={len(train_rows)} strong_rows={train_strong} "
        f"strong_rate={train_strong / max(len(train_rows), 1):.3f} "
        f"weak_positive_weight={args.weak_positive_weight}"
    )
    print(
        f"[label] val_rows={len(val_rows)} strong_rows={val_strong} "
        f"strong_rate={val_strong / max(len(val_rows), 1):.3f}"
    )
    train_ds = GraphMatchDataset(train_rows, sbert=None)
    val_ds = GraphMatchDataset(val_rows, sbert=None)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )
    model = QueryGNN(
        in_dim=384, hidden_dim=args.hidden, num_layers=args.num_layers, query_dim=384
    ).to(device)
    if args.resume_checkpoint:
        state = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[info] loaded resume checkpoint: {args.resume_checkpoint}")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sbert = SentenceTransformer(args.sbert, device=device)
    val_evalable = [s for s in val_s if s.turns and s.evidence_global_ids and s.answer]
    val_subset = stratified_sample(val_evalable, args.val_limit, seed=args.val_seed)
    val_graphs = []
    for s in val_subset:
        g = build_sentence_graph(s.turns, topk=args.topk_graph, sbert_model=args.sbert)
        td = g.to_torch(device="cpu")
        val_graphs.append(
            {
                "sample": s,
                "question": s.question,
                "x": td["x"],
                "edge_index": td["edge_index"],
                "turn_meta": _build_turn_meta(s),
            }
        )
    reader = None
    judge = None
    if not args.skip_qa_eval:
        reader = QwenReader(model_name=args.reader_model, backend=args.reader_backend)
        judge = LLMJudge(backend=args.judge_backend, cache_dir=args.judge_cache_dir)
    best_metric = -1.0
    best_epoch = -1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_preview = Path(args.checkpoint_pattern.format(ep=1))
    ckpt_preview.parent.mkdir(parents=True, exist_ok=True)
    t_train0 = time.time()
    hist: list[dict] = []
    max_epochs = args.epochs
    epoch = 1
    while epoch <= max_epochs:
        loss = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            lam=args.lam,
            neg_ratio=args.neg_ratio,
            rank_lam=args.rank_lam,
            rank_margin=args.rank_margin,
            hard_negatives=args.hard_negatives,
        )
        r5 = eval_recall(model, val_loader, device, k=5)
        strong_h5 = eval_strong_hit(model, val_loader, device, k=5)
        if args.skip_qa_eval:
            val_acc, val_sess_r5 = float("nan"), float("nan")
        else:
            assert reader is not None and judge is not None
            val_acc, val_sess_r5 = eval_qa_accuracy(
                model=model,
                val_graphs=val_graphs,
                sbert=sbert,
                reader=reader,
                judge=judge,
                device=device,
                k=5,
                reader_batch_size=args.reader_batch_size,
            )
        metric = r5 if args.skip_qa_eval else val_acc
        if np.isnan(metric):
            tag = ""
        elif metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            tag = "  (best so far)"
        else:
            tag = ""
        ep_path = Path(args.checkpoint_pattern.format(ep=epoch))
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ep_path)
        print(
            f"[ep {epoch}/{max_epochs}] train_loss={loss:.3f}  val_R@5={r5:.3f}  "
            f"val_StrongHit@5={strong_h5:.3f}  val_acc={val_acc:.3f}{tag}"
        )
        hist.append(
            {
                "epoch": epoch,
                "train_loss": float(loss),
                "val_R5": float(r5),
                "val_StrongHit5": float(strong_h5),
                "val_acc": float(val_acc),
                "qa_eval": True,
            }
        )
        if epoch == args.epochs and best_epoch == args.epochs and max_epochs < 8:
            max_epochs = min(8, args.epochs + 3)
            print(f"[info] epoch {args.epochs} is current best; extending training to {max_epochs} epochs.")
        epoch += 1

    print(f"BEST_EPOCH_BY_METRIC: epoch {best_epoch}, metric={best_metric:.3f}")
    curve_path = Path(args.curve_out)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(
        json.dumps(
            [
                {
                    "epoch": r["epoch"],
                    "train_loss": r["train_loss"],
                    "val_R5": r["val_R5"],
                    "val_StrongHit5": r["val_StrongHit5"],
                    "val_acc": r["val_acc"],
                }
                for r in hist
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] saved training curve: {curve_path}")
    print(f"training_wall_clock_sec={time.time() - t_train0:.2f}")
    if judge is not None:
        js = judge.stats()
        print(
            f"judge_calls={js['total_judge_calls']} judge_cache_hit_rate={js['judge_cache_hit_rate']:.3f} "
            f"judge_est_cost_cny={js['est_cost_cny']}"
        )
    print("\n=== TRAINING CURVE (all epochs) ===")
    print("epoch\tloss\tval_R@5\tval_StrongHit@5\tval_acc")
    for row in hist:
        val_acc_txt = "nan" if row["val_acc"] is None else f"{row['val_acc']:.3f}"
        print(
            f"{row['epoch']}\t{row['train_loss']:.3f}\t{row['val_R5']:.3f}\t"
            f"{row['val_StrongHit5']:.3f}\t{val_acc_txt}"
        )
    best_acc = max((float(r["val_acc"]) for r in hist if r["val_acc"] is not None), default=float("nan"))
    if args.skip_qa_eval:
        return
    if np.isnan(best_acc) or best_acc < 0.45:
        print("[diag] GNN val_acc < 0.45")
        for row in hist:
            print(
                f"  ep{row['epoch']}: train_loss={row['train_loss']:.3f} val_R@5={row['val_R5']:.3f} "
                f"val_acc={row['val_acc'] if row['qa_eval'] else float('nan'):.3f}"
            )
        r5_improved = hist[-1]["val_R5"] > hist[0]["val_R5"]
        acc_valid = [h["val_acc"] for h in hist if h["qa_eval"] and not np.isnan(h["val_acc"])]
        acc_improved = len(acc_valid) >= 2 and acc_valid[-1] > acc_valid[0]
        if r5_improved and not acc_improved:
            print("[diag] val_R@5 improved but val_acc did not -> retrieval gains may not propagate to QA.")
        elif not r5_improved:
            print("[diag] val_R@5 did not improve -> training signal may be weak.")


if __name__ == "__main__":
    main()
