from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import torch
from collections import defaultdict
from pathlib import Path

from grace.datasets.longmemeval import LongMemSample, load_longmemeval_s
from grace.eval.retrieval_metrics import hit_at_k, mrr, recall_at_k, session_recall_at_k
from grace.graphs.build import build_sentence_graph
from grace.qa.judge import LLMJudge, PRICE_PER_1M_CNY
from grace.qa.reader import QwenReader
from grace.retrievers.bm25 import BM25Retriever
from grace.retrievers.gnn import GNNRetriever
from grace.retrievers.ppr import PPRRetriever
from grace.retrievers.sbert import SBERTRetriever
from tqdm import tqdm


def stratified_sample(samples: list[LongMemSample], limit: int, seed: int = 42) -> list[LongMemSample]:
    by_type: dict[str, list[LongMemSample]] = defaultdict(list)
    for s in samples:
        by_type[s.question_type].append(s)
    rng = random.Random(seed)
    for rows in by_type.values():
        rng.shuffle(rows)
    types = sorted(by_type.keys())
    if not types:
        return []
    per_type = max(1, limit // len(types))
    picked: list[LongMemSample] = []
    leftovers: list[LongMemSample] = []
    for t in types:
        items = by_type[t]
        picked.extend(items[:per_type])
        leftovers.extend(items[per_type:])
    if len(picked) < limit:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: limit - len(picked)])
    rng.shuffle(picked)
    return picked[:limit]


def _resolve_local_sbert_model(model_name: str) -> str:
    cache_root = Path.home() / ".cache/huggingface/hub"
    safe_name = model_name.replace("/", "--")
    snaps_dir = cache_root / f"models--{safe_name}" / "snapshots"
    snaps = sorted([p for p in snaps_dir.iterdir() if p.is_dir()])
    if not snaps:
        raise FileNotFoundError(f"No local snapshot for {model_name}")
    return str(snaps[-1])


def _turn_content(turn: dict) -> str:
    for key in ("content", "text", "value", "message", "body"):
        v = turn.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return str(turn)


def _build_turn_meta(sample: LongMemSample) -> dict[int, dict]:
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


def _retrieve_one(
    s: LongMemSample,
    retriever: str,
    k: int,
    graph_topk: int,
    sbert_model: str,
    gnn_retriever: GNNRetriever | None = None,
    rerank_pool: int = 20,
) -> tuple[list[int], list[dict]]:
    texts = [t.text for t in s.turns]
    if retriever == "bm25":
        r = BM25Retriever(texts).retrieve(s.question, k=k)
    elif retriever == "sbert":
        r = SBERTRetriever(texts, model_name=sbert_model).retrieve(s.question, k=k)
    elif retriever == "ppr":
        g = build_sentence_graph(s.turns, topk=graph_topk, sbert_model=sbert_model)
        r = PPRRetriever(g, sbert_model=sbert_model).retrieve(s.question, k=k)
    elif retriever == "gnn":
        if gnn_retriever is None:
            raise ValueError("GNN retriever selected but no checkpoint-loaded retriever was provided.")
        g = build_sentence_graph(s.turns, topk=graph_topk, sbert_model=sbert_model)
        r = gnn_retriever.retrieve(s.question, g.to_torch(device=gnn_retriever.device), k=k)
    elif retriever == "gnn+sbert_rerank":
        if gnn_retriever is None:
            raise ValueError("gnn+sbert_rerank selected but no checkpoint-loaded retriever was provided.")
        g = build_sentence_graph(s.turns, topk=graph_topk, sbert_model=sbert_model)
        pool_k = max(k, rerank_pool)
        gnn_ranked = gnn_retriever.retrieve(
            s.question, g.to_torch(device=gnn_retriever.device), k=pool_k
        )
        cand_ids = [i for i, _, _ in gnn_ranked]
        cand_texts = [g.node_texts[i] for i in cand_ids]
        sb_ranked = SBERTRetriever(cand_texts, model_name=sbert_model).retrieve(s.question, k=k)
        r = [(cand_ids[i], score, text) for i, score, text in sb_ranked]
    else:
        raise ValueError(f"Unsupported retriever: {retriever}")
    ids = [i for i, _, _ in r]
    tm = _build_turn_meta(s)
    ctx = [tm.get(i, {"session_id": str(s.turn_to_session.get(i, "unknown")), "session_date": "", "role": "unknown", "content": s.turns[i].text if 0 <= i < len(s.turns) else ""}) for i in ids]
    return ids, ctx


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end retrieval + QA + judge evaluation.")
    p.add_argument("--data", default="data/raw/longmemeval/longmemeval_s.json")
    p.add_argument("--retriever", choices=("bm25", "sbert", "ppr", "gnn", "gnn+sbert_rerank"), default="sbert")
    p.add_argument("--checkpoint", default="", help="Required when --retriever gnn")
    p.add_argument("--rerank-pool", type=int, default=20)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graph-topk", type=int, default=5)
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sbert-local-only", action="store_true")
    p.add_argument("--reader-backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--reader-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--reader-batch-size", type=int, default=16)
    p.add_argument("--judge-backend", choices=("siliconflow", "deepseek", "local_vllm"), default="siliconflow")
    p.add_argument("--judge-concurrency", type=int, default=10)
    p.add_argument("--judge-cache-dir", default="data/processed/judge_cache")
    p.add_argument("--judge-max-retries", type=int, default=3)
    p.add_argument("--output", default="experiments/results/full_eval.json")
    args = p.parse_args()

    t0 = time.time()
    all_samples = load_longmemeval_s(args.data)
    evalable = [s for s in all_samples if s.turns and s.evidence_global_ids and s.answer]
    subset = stratified_sample(evalable, args.limit, seed=args.seed)
    if not subset:
        raise SystemExit("No evaluable samples.")
    est_judge_tokens = len(subset) * 350
    est_judge_cost = (est_judge_tokens / 1_000_000.0) * PRICE_PER_1M_CNY[args.judge_backend]["strong"]
    print(
        f"[cost-estimate] retriever={args.retriever} n={len(subset)} "
        f"judge_upper_bound_cny={est_judge_cost:.4f} (assume 350 tokens/sample, no cache)"
    )

    sbert_model = args.sbert_model
    if args.sbert_local_only:
        sbert_model = _resolve_local_sbert_model(args.sbert_model)
    gnn_retriever: GNNRetriever | None = None
    if args.retriever in ("gnn", "gnn+sbert_rerank"):
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when --retriever gnn")
        if args.checkpoint == "random_init":
            from grace.models.query_gnn import QueryGNN
            from sentence_transformers import SentenceTransformer
            torch.manual_seed(42)
            class _RandomInitGNNRetriever:
                def __init__(self, sbert_name: str):
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.sbert = SentenceTransformer(sbert_name, device=self.device)
                    self.model = QueryGNN(in_dim=384, hidden_dim=256, num_layers=2, query_dim=384).to(self.device)
                    self.model.eval()
                @torch.no_grad()
                def retrieve(self, query: str, graph_data: dict, k: int = 5):
                    q = self.sbert.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
                    q_t = torch.tensor(q, device=self.device, dtype=torch.float32)
                    x = graph_data["x"].to(self.device)
                    ei = graph_data["edge_index"].to(self.device)
                    logits = self.model(x, ei, q_t)
                    kk = min(k, int(logits.size(0)))
                    topk = torch.topk(logits, k=kk)
                    texts = graph_data["node_texts"]
                    return [(int(i), float(v), texts[int(i)]) for i, v in zip(topk.indices.tolist(), topk.values.tolist())]
            print("[sanity] ckpt path: random_init")
            gnn_retriever = _RandomInitGNNRetriever(sbert_model)
            fp = next(gnn_retriever.model.parameters())
            print(f"[sanity] random-init model first_param sum={fp.float().sum().item():.4f}")
        else:
            state = torch.load(args.checkpoint, map_location="cpu")
            keys = list(state.keys()) if isinstance(state, dict) else []
            print(f"[sanity] ckpt path: {args.checkpoint}")
            print(f"[sanity] ckpt keys (first 5): {keys[:5]}")
            if keys:
                w = state[keys[0]]
                print(f"[sanity] {keys[0]} shape={tuple(w.shape)} sum={w.float().sum().item():.4f} mean={w.float().mean().item():.6f}")
            gnn_retriever = GNNRetriever(
                model_path=args.checkpoint,
                sbert_name=sbert_model,
            )
            first_param = next(gnn_retriever.model.parameters())
            print(f"[sanity] loaded model first_param sum={first_param.float().sum().item():.4f}")

    t_retr_start = time.time()
    retrieved_ids: list[list[int]] = []
    contexts: list[list[dict]] = []
    for s in tqdm(subset, desc=f"[{args.retriever}] retrieve", unit="q"):
        ids, ctx = _retrieve_one(
            s,
            args.retriever,
            args.k,
            args.graph_topk,
            sbert_model,
            gnn_retriever=gnn_retriever,
            rerank_pool=args.rerank_pool,
        )
        retrieved_ids.append(ids)
        contexts.append(ctx)
    t_retr = time.time() - t_retr_start

    t_reader_start = time.time()
    reader = QwenReader(model_name=args.reader_model, backend=args.reader_backend)
    if subset:
        sample_prompt = reader._build_prompt(
            question=subset[0].question,
            context_text=reader._format_context(contexts[0]),
        )
        print("=== SAMPLE CHAT-TEMPLATED PROMPT (#0) ===")
        print(sample_prompt)
    questions = [s.question for s in subset]
    preds: list[str] = []
    bs = max(1, args.reader_batch_size)
    for i in tqdm(range(0, len(questions), bs), desc=f"[{args.retriever}] reader", unit="batch"):
        preds.extend(reader.answer_batch(questions[i : i + bs], contexts[i : i + bs]))
    t_reader = time.time() - t_reader_start

    t_judge_start = time.time()
    judge = LLMJudge(
        backend=args.judge_backend,
        cache_dir=args.judge_cache_dir,
        max_retries=args.judge_max_retries,
    )
    judge_samples = [
        {
            "question": s.question,
            "gold": s.answer or "",
            "predicted": ptext,
            "question_type": s.question_type,
        }
        for s, ptext in zip(subset, preds)
    ]
    judge_out = asyncio.run(judge.judge_batch(judge_samples, concurrency=args.judge_concurrency))
    t_judge = time.time() - t_judge_start

    per_question = []
    by_type_acc: dict[str, list[float]] = defaultdict(list)
    by_type_sessr: dict[str, list[float]] = defaultdict(list)
    all_acc: list[float] = []
    all_r: list[float] = []
    all_h: list[float] = []
    all_sr: list[float] = []
    all_mrr: list[float] = []

    for s, rid, pred, jo in zip(subset, retrieved_ids, preds, judge_out):
        pos = set(s.evidence_global_ids)
        sr = session_recall_at_k(rid, s.turn_to_session, s.gold_sessions, args.k)
        rr = recall_at_k(rid, pos, args.k)
        hh = hit_at_k(rid, pos, args.k)
        mr = mrr(rid, pos)
        correct = bool(jo.get("correct", False))
        all_acc.append(1.0 if correct else 0.0)
        all_r.append(rr)
        all_h.append(hh)
        all_sr.append(sr)
        all_mrr.append(mr)
        by_type_acc[s.question_type].append(1.0 if correct else 0.0)
        by_type_sessr[s.question_type].append(sr)
        per_question.append(
            {
                "qid": s.question_id,
                "question_type": s.question_type,
                "question": s.question,
                "gold": s.answer,
                "predicted": pred,
                "correct": correct,
                "retrieved_turn_ids": rid,
                "retrieved_sessions": [
                    s.turn_to_session[i] for i in rid if i in s.turn_to_session
                ],
                "sess_recall_at_5": sr,
                "judge_reasoning": jo.get("reasoning", ""),
            }
        )

    def _mean(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    by_type = {}
    for t in sorted(by_type_acc):
        by_type[t] = {
            "accuracy": _mean(by_type_acc[t]),
            "sess_recall_at_5": _mean(by_type_sessr[t]),
            "n": len(by_type_acc[t]),
        }

    out_obj = {
        "config": {
            "retriever": args.retriever,
            "limit": args.limit,
            "k": args.k,
            "seed": args.seed,
            "reader_backend": args.reader_backend,
            "reader_model": args.reader_model,
            "judge_backend": args.judge_backend,
        },
        "overall": {
            "accuracy": _mean(all_acc),
            "sess_recall_at_5": _mean(all_sr),
            "recall_at_5": _mean(all_r),
            "hit_at_5": _mean(all_h),
            "mrr": _mean(all_mrr),
            "n": len(subset),
        },
        "by_type": by_type,
        "per_question": per_question,
        "stats": {
            **judge.stats(),
            "avg_predicted_words": _mean([len(str(p).split()) for p in preds]),
            "retrieval_latency_sec": round(t_retr, 4),
            "reader_latency_sec": round(t_reader, 4),
            "judge_latency_sec": round(t_judge, 4),
            "total_latency_sec": round(time.time() - t0, 4),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved: {out_path}")
    print(
        f"overall acc={out_obj['overall']['accuracy']:.3f} "
        f"sessR@5={out_obj['overall']['sess_recall_at_5']:.3f} n={len(subset)}"
    )
    print(
        f"judge cache hit={out_obj['stats']['judge_cache_hit_rate']:.3f} "
        f"est_cost_cny={out_obj['stats']['est_cost_cny']}"
    )


if __name__ == "__main__":
    main()
