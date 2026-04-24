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


def train_one_epoch(model, loader, opt, device, lam: float) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        opt.zero_grad()
        loss = 0.0
        m = 0
        for x, edge_index, q, pos_mask, _ in batch:
            x, edge_index = x.to(device), edge_index.to(device)
            q, pos_mask = q.to(device), pos_mask.to(device)
            logits = model(x, edge_index, q)
            loss = loss + combined_loss(logits, pos_mask, lam=lam)
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
        for x, edge_index, q, pos_mask, _ in batch:
            x, edge_index = x.to(device), edge_index.to(device)
            q, pos_mask = q.to(device), pos_mask.to(device)
            logits = model(x, edge_index, q)
            kk = min(k, int(logits.size(0)))
            topk = torch.topk(logits, k=kk).indices
            hit = pos_mask[topk].any().item()
            hits += int(hit)
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
    p.add_argument("--out", default="experiments/checkpoints/gnn.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk_graph", type=int, default=5)
    p.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--val_limit", type=int, default=30)
    p.add_argument("--val_seed", type=int, default=42)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--reader_backend", choices=("vllm", "transformers"), default="transformers")
    p.add_argument("--reader_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--reader_batch_size", type=int, default=2)
    p.add_argument("--judge_backend", choices=("siliconflow", "deepseek", "local_vllm"), default="siliconflow")
    p.add_argument("--judge_cache_dir", default="data/processed/judge_cache")
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
    train_rows = longmem_samples_to_graph_rows(
        train_s, topk=args.topk_graph, sbert=args.sbert
    )
    val_rows = longmem_samples_to_graph_rows(val_s, topk=args.topk_graph, sbert=args.sbert)
    if not train_rows:
        print("No training rows (missing evidence or empty turns).")
        return
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
    reader = QwenReader(model_name=args.reader_model, backend=args.reader_backend)
    judge = LLMJudge(backend=args.judge_backend, cache_dir=args.judge_cache_dir)
    best_metric = -1.0
    best_epoch = -1
    best_acc = 0.0
    best_path = Path("experiments/checkpoints/gnn_best.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_train0 = time.time()
    proxy_mode = False
    hist: list[dict] = []
    max_epochs = args.epochs
    epoch = 1
    while epoch <= max_epochs:
        loss = train_one_epoch(model, train_loader, opt, device, lam=args.lam)
        r5 = eval_recall(model, val_loader, device, k=5)
        val_acc = float("nan")
        qa_eval_start = time.time()
        qa_evaluated = False
        if not proxy_mode:
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
            qa_eval_elapsed = time.time() - qa_eval_start
            qa_evaluated = True
            if qa_eval_elapsed > 300.0 and epoch < max_epochs:
                proxy_mode = True
                print("[warn] Per-epoch QA eval exceeded 5 min; switching to Recall@5 proxy and evaluating QA on final epoch only.")
        else:
            val_sess_r5 = float("nan")
        metric = val_acc if qa_evaluated else r5
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_acc = val_acc if qa_evaluated else best_acc
            torch.save(model.state_dict(), best_path)
            tag = "  (best so far)"
        else:
            tag = ""
        if args.save_every_epoch:
            ep_path = Path(str(args.out).format(ep=epoch))
            ep_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ep_path)
        print(
            f"[ep {epoch}/{max_epochs}] train_loss={loss:.3f}  val_R@5={r5:.3f}  "
            f"val_acc={val_acc if qa_evaluated else float('nan'):.3f}{tag}"
        )
        hist.append(
            {
                "epoch": epoch,
                "train_loss": float(loss),
                "val_R5": float(r5),
                "val_acc": float(val_acc) if qa_evaluated else None,
                "qa_eval": bool(qa_evaluated),
            }
        )
        if epoch == args.epochs and best_epoch == args.epochs and max_epochs < 8:
            max_epochs = min(8, args.epochs + 3)
            print(f"[info] epoch {args.epochs} is current best; extending training to {max_epochs} epochs.")
        epoch += 1

    if proxy_mode:
        print("[warn] Best checkpoint selected by Recall@5 proxy due to QA eval cost.")
        final_acc, final_sess_r5 = eval_qa_accuracy(
            model=model,
            val_graphs=val_graphs,
            sbert=sbert,
            reader=reader,
            judge=judge,
            device=device,
            k=5,
            reader_batch_size=args.reader_batch_size,
        )
        print(f"[final QA] val_acc={final_acc:.3f} val_sessR@5={final_sess_r5:.3f}")
    print(
        f"BEST: epoch {best_epoch}, val_acc={best_acc:.3f}, saved to {best_path}"
    )
    curve_path = Path(args.curve_out)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(
        json.dumps(
            [
                {
                    "epoch": r["epoch"],
                    "train_loss": r["train_loss"],
                    "val_R5": r["val_R5"],
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
    js = judge.stats()
    print(
        f"judge_calls={js['total_judge_calls']} judge_cache_hit_rate={js['judge_cache_hit_rate']:.3f} "
        f"judge_est_cost_cny={js['est_cost_cny']}"
    )
    if best_acc < 0.45:
        print("[diag] GNN val_acc < 0.45")
        for row in hist:
            print(
                f"  ep{row['epoch']}: train_loss={row['train_loss']:.3f} val_R@5={row['val_R5']:.3f} "
                f"val_acc={row['val_acc'] if row['qa_eval'] else float('nan'):.3f}"
            )
        r5_improved = hist[-1]["val_r5"] > hist[0]["val_r5"]
        acc_valid = [h["val_acc"] for h in hist if h["qa_eval"] and not np.isnan(h["val_acc"])]
        acc_improved = len(acc_valid) >= 2 and acc_valid[-1] > acc_valid[0]
        if r5_improved and not acc_improved:
            print("[diag] val_R@5 improved but val_acc did not -> retrieval gains may not propagate to QA.")
        elif not r5_improved:
            print("[diag] val_R@5 did not improve -> training signal may be weak.")


if __name__ == "__main__":
    main()
