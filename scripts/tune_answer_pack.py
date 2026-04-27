from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from grace.datasets.longmemeval import load_longmemeval_s
from grace.eval.retrieval_metrics import hit_at_k, mrr, recall_at_k, session_recall_at_k
from grace.graphs.build import build_sentence_graph
from grace.retrievers.bm25 import BM25Retriever
from grace.retrievers.gnn import GNNRetriever
from grace.retrievers.sbert import SBERTRetriever
from run_full_eval import (
    _answer_in_context,
    _build_turn_meta,
    _minmax,
    _resolve_local_sbert_model,
    _select_answer_aware_context,
    _select_diverse_by_session,
    stratified_sample,
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _fused_candidates(
    sample,
    gnn_retriever: GNNRetriever,
    sbert_model: str,
    graph_topk: int,
    pool_k: int,
    fusion_gnn_weight: float,
    fusion_sbert_weight: float,
    fusion_bm25_weight: float,
) -> list[tuple[int, float, str]]:
    g = build_sentence_graph(sample.turns, topk=graph_topk, sbert_model=sbert_model)
    gnn_ranked = gnn_retriever.retrieve(
        sample.question,
        g.to_torch(device=gnn_retriever.device),
        k=pool_k,
    )
    cand_ids = [i for i, _, _ in gnn_ranked]
    cand_texts = [g.node_texts[i] for i in cand_ids]

    gnn_scores = _minmax([float(score) for _, score, _ in gnn_ranked])

    sbert_scores = [0.0 for _ in cand_ids]
    for local_i, score, _ in SBERTRetriever(cand_texts, model_name=sbert_model).retrieve(
        sample.question, k=len(cand_ids)
    ):
        sbert_scores[local_i] = float(score)
    sbert_scores = _minmax(sbert_scores)

    bm25_scores = [0.0 for _ in cand_ids]
    for local_i, score, _ in BM25Retriever(cand_texts).retrieve(sample.question, k=len(cand_ids)):
        bm25_scores[local_i] = float(score)
    bm25_scores = _minmax(bm25_scores)

    fused: list[tuple[int, float, str]] = []
    for local_i, gid in enumerate(cand_ids):
        score = (
            fusion_gnn_weight * gnn_scores[local_i]
            + fusion_sbert_weight * sbert_scores[local_i]
            + fusion_bm25_weight * bm25_scores[local_i]
        )
        fused.append((gid, score, g.node_texts[gid]))
    return sorted(fused, key=lambda x: x[1], reverse=True)


def _metrics(samples, selected_ids: list[list[int]]) -> dict:
    recalls: list[float] = []
    hits: list[float] = []
    mrrs: list[float] = []
    sessrs: list[float] = []
    answer_ctxs: list[float] = []
    for s, ids in zip(samples, selected_ids):
        pos = set(s.evidence_global_ids)
        tm = _build_turn_meta(s)
        ctx = [
            tm.get(
                i,
                {
                    "session_id": str(s.turn_to_session.get(i, "unknown")),
                    "session_date": "",
                    "role": "unknown",
                    "content": s.turns[i].text if 0 <= i < len(s.turns) else "",
                },
            )
            for i in ids
        ]
        recalls.append(recall_at_k(ids, pos, len(ids)))
        hits.append(hit_at_k(ids, pos, len(ids)))
        mrrs.append(mrr(ids, pos))
        sessrs.append(session_recall_at_k(ids, s.turn_to_session, s.gold_sessions, len(ids)))
        answer_ctxs.append(1.0 if _answer_in_context(s.answer, ctx) else 0.0)
    return {
        "recall_at_k": _mean(recalls),
        "hit_at_k": _mean(hits),
        "mrr": _mean(mrrs),
        "sess_recall_at_k": _mean(sessrs),
        "answer_in_context_at_k": _mean(answer_ctxs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Tune final context packing without running reader/judge.")
    p.add_argument("--data", default="data/raw/longmemeval/longmemeval_s.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gnn-arch", choices=("sage", "sage_res", "sage_skip", "sage_qa", "gat"), default="sage")
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--graph-topk", type=int, default=5)
    p.add_argument("--rerank-pool", type=int, default=20)
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sbert-local-only", action="store_true")
    p.add_argument("--fusion-gnn-weight", type=float, default=0.5)
    p.add_argument("--fusion-sbert-weight", type=float, default=0.3)
    p.add_argument("--fusion-bm25-weight", type=float, default=0.2)
    p.add_argument("--pack-weights", default="0.10,0.20,0.35,0.50,0.75")
    p.add_argument("--pack-session-penalties", default="0.00,0.03,0.05,0.08")
    p.add_argument("--diverse-penalties", default="0.00,0.05,0.10,0.15")
    p.add_argument("--output", default="experiments/results/answer_pack_tuning.json")
    args = p.parse_args()

    sbert_model = _resolve_local_sbert_model(args.sbert_model) if args.sbert_local_only else args.sbert_model
    samples = load_longmemeval_s(args.data)
    evalable = [s for s in samples if s.turns and s.evidence_global_ids and s.answer]
    subset = stratified_sample(evalable, args.limit, seed=args.seed)
    gnn_retriever = GNNRetriever(
        model_path=args.checkpoint,
        sbert_name=sbert_model,
        arch=args.gnn_arch,
        gat_heads=args.gat_heads,
    )

    ranked_by_sample = []
    for s in tqdm(subset, desc="[tune] fused candidates", unit="q"):
        ranked_by_sample.append(
            _fused_candidates(
                s,
                gnn_retriever=gnn_retriever,
                sbert_model=sbert_model,
                graph_topk=args.graph_topk,
                pool_k=max(args.k, args.rerank_pool),
                fusion_gnn_weight=args.fusion_gnn_weight,
                fusion_sbert_weight=args.fusion_sbert_weight,
                fusion_bm25_weight=args.fusion_bm25_weight,
            )
        )

    rows = []
    baseline_ids = [[gid for gid, _, _ in ranked[: args.k]] for ranked in ranked_by_sample]
    rows.append({"method": "fusion_topk", **_metrics(subset, baseline_ids)})
    pool_ids = [[gid for gid, _, _ in ranked] for ranked in ranked_by_sample]
    rows.append({"method": f"fusion_pool_top{max(args.k, args.rerank_pool)}", **_metrics(subset, pool_ids)})

    for penalty in [float(x) for x in args.diverse_penalties.split(",") if x.strip()]:
        ids = [
            [gid for gid, _, _ in _select_diverse_by_session(ranked, s.turn_to_session, args.k, penalty)]
            for s, ranked in zip(subset, ranked_by_sample)
        ]
        rows.append({"method": "fusion_diverse", "session_penalty": penalty, **_metrics(subset, ids)})

    for weight in [float(x) for x in args.pack_weights.split(",") if x.strip()]:
        for penalty in [float(x) for x in args.pack_session_penalties.split(",") if x.strip()]:
            ids = [
                [
                    gid
                    for gid, _, _ in _select_answer_aware_context(
                        s.question,
                        ranked,
                        s.turn_to_session,
                        args.k,
                        pack_weight=weight,
                        session_penalty=penalty,
                    )
                ]
                for s, ranked in zip(subset, ranked_by_sample)
            ]
            rows.append(
                {
                    "method": "fusion_answer_pack",
                    "pack_weight": weight,
                    "session_penalty": penalty,
                    **_metrics(subset, ids),
                }
            )

    rows = sorted(rows, key=lambda r: (r["answer_in_context_at_k"], r["sess_recall_at_k"], r["hit_at_k"]), reverse=True)
    out = {
        "config": vars(args),
        "n": len(subset),
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved: {out_path}")
    for row in rows[:12]:
        print(row)


if __name__ == "__main__":
    main()
