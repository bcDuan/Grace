"""Evaluate BM25 / SBERT / PPR on LongMemEval with multi-granularity metrics."""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_ROOT / "src"))

from grace.datasets.longmemeval import LongMemSample, load_longmemeval_s  # noqa: E402
from grace.eval.retrieval_metrics import (  # noqa: E402
    hit_at_k,
    mrr,
    recall_at_k,
    session_recall_at_k,
)
from grace.graphs.build import build_sentence_graph  # noqa: E402
from grace.retrievers.bm25 import BM25Retriever  # noqa: E402
from grace.retrievers.ppr import PPRRetriever  # noqa: E402
from grace.retrievers.sbert import SBERTRetriever  # noqa: E402


def _evaluable(samples: list[LongMemSample]) -> list[LongMemSample]:
    return [s for s in samples if s.turns and s.evidence_global_ids and s.gold_sessions]


def stratified_sample(samples: list[LongMemSample], limit: int, seed: int = 42) -> list[LongMemSample]:
    """Sample evenly across question_type, up to limit total."""
    if limit <= 0:
        return []
    by_type: dict[str, list[LongMemSample]] = defaultdict(list)
    for s in samples:
        by_type[s.question_type].append(s)
    if not by_type:
        return []

    rng = random.Random(seed)
    for items in by_type.values():
        rng.shuffle(items)

    type_keys = sorted(by_type.keys())
    per_type = max(1, limit // len(type_keys))
    picked: list[LongMemSample] = []
    leftovers: list[LongMemSample] = []

    for t in type_keys:
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
    model_root = cache_root / f"models--{safe_name}" / "snapshots"
    if not model_root.exists():
        raise FileNotFoundError(f"Local cache not found for model: {model_name}")
    snaps = sorted([p for p in model_root.iterdir() if p.is_dir()])
    if not snaps:
        raise FileNotFoundError(f"No local snapshot found for model: {model_name}")
    return str(snaps[-1])


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def main() -> None:
    p = argparse.ArgumentParser(description="LongMemEval retrieval metrics.")
    p.add_argument("--data", default=str(_ROOT / "data/raw/longmemeval/longmemeval_s.json"))
    p.add_argument("--limit", type=int, default=50, help="Stratified sample size after evaluable filter.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--top_k", type=int, default=None, help="Alias for --k.")
    p.add_argument("--graph-topk", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", choices=("bm25", "sbert", "ppr", "all"), default="all")
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sbert-local-only", action="store_true")
    p.add_argument("--progress-every", type=int, default=5)
    args = p.parse_args()

    k = args.top_k if args.top_k is not None else args.k
    t0 = time.time()
    print(f"[INFO] Loading data: {args.data}")
    all_samples = load_longmemeval_s(args.data)
    ev_samples = _evaluable(all_samples)
    print(f"[INFO] Loaded raw={len(all_samples)} evaluable={len(ev_samples)}")

    limit = min(args.limit, len(ev_samples))
    subset = stratified_sample(ev_samples, limit=limit, seed=args.seed)
    if not subset:
        print("No evaluable samples (need turns + evidence_global_ids + gold_sessions).")
        return

    type_counts: dict[str, int] = defaultdict(int)
    for s in subset:
        type_counts[s.question_type] += 1

    print(f"[INFO] Stratified sample n={len(subset)} seed={args.seed} types={len(type_counts)}")
    print("[INFO] Type counts:")
    for t in sorted(type_counts):
        print(f"  - {t}: {type_counts[t]}")

    methods = [args.method] if args.method != "all" else ["bm25", "sbert", "ppr"]
    metrics: dict[str, dict[str, list[float]]] = {
        m: {"R": [], "Hit": [], "SessR": [], "MRR": []} for m in methods
    }
    by_type_sessr: dict[str, dict[str, list[float]]] = {
        t: {m: [] for m in methods} for t in type_counts
    }

    sbert_st = None
    if "sbert" in methods:
        from sentence_transformers import SentenceTransformer

        model_ref = args.sbert_model
        if args.sbert_local_only:
            model_ref = _resolve_local_sbert_model(args.sbert_model)
            print(f"[INFO] Using local SBERT snapshot: {model_ref}")
        print(f"[INFO] Loading SBERT model: {args.sbert_model}")
        sbert_st = SentenceTransformer(model_ref)
        print("[INFO] SBERT model loaded")

    for i, s in enumerate(subset, start=1):
        texts = [t.text for t in s.turns]
        pos = set(s.evidence_global_ids)
        gold_sessions = set(s.gold_sessions)

        ranked_by_method: dict[str, list[int]] = {}

        if "bm25" in methods:
            base = BM25Retriever(texts)
            ranked_by_method["bm25"] = [idx for idx, _, _ in base.retrieve(s.question, k=k)]

        if "sbert" in methods:
            sbr = SBERTRetriever(texts, model=sbert_st)
            ranked_by_method["sbert"] = [idx for idx, _, _ in sbr.retrieve(s.question, k=k)]

        if "ppr" in methods:
            g = build_sentence_graph(s.turns, topk=args.graph_topk)
            pr = PPRRetriever(g)
            ranked_by_method["ppr"] = [idx for idx, _, _ in pr.retrieve(s.question, k=k)]

        for m in methods:
            ranked = ranked_by_method[m]
            r = recall_at_k(ranked, pos, k)
            h = hit_at_k(ranked, pos, k)
            sr = session_recall_at_k(ranked, s.turn_to_session, gold_sessions, k)
            rr = mrr(ranked, pos)
            metrics[m]["R"].append(r)
            metrics[m]["Hit"].append(h)
            metrics[m]["SessR"].append(sr)
            metrics[m]["MRR"].append(rr)
            by_type_sessr[s.question_type][m].append(sr)

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(subset)):
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0.0
            eta = (len(subset) - i) / speed if speed > 0 else 0.0
            parts = [f"[PROGRESS] {i}/{len(subset)}", f"elapsed={elapsed:.1f}s", f"eta={eta:.1f}s"]
            for m in methods:
                parts.append(f"{m.upper()} SessR={_mean(metrics[m]['SessR']):.3f}")
            print("  ".join(parts))

    print()
    print(f"n={len(subset)}  (stratified across {len(type_counts)} question types)")
    print("method  R@5    Hit@5   SessR@5   MRR")
    print("------  -----  -----   -------   -----")
    for m in methods:
        print(
            f"{m.upper():<6}  {_mean(metrics[m]['R']):.3f}  {_mean(metrics[m]['Hit']):.3f}   "
            f"{_mean(metrics[m]['SessR']):.3f}     {_mean(metrics[m]['MRR']):.3f}"
        )

    if len(methods) == 3:
        print("By question_type (SessR@5):")
        for t in sorted(type_counts):
            print(
                f"{t:<24} BM25={_mean(by_type_sessr[t]['bm25']):.3f}  "
                f"SBERT={_mean(by_type_sessr[t]['sbert']):.3f}  "
                f"PPR={_mean(by_type_sessr[t]['ppr']):.3f}  (n={type_counts[t]})"
            )

        m_b = _mean(metrics["bm25"]["SessR"])
        m_s = _mean(metrics["sbert"]["SessR"])
        m_p = _mean(metrics["ppr"]["SessR"])
        print("Checks:")
        print(f"  [ {'PASS' if _mean(metrics['bm25']['Hit']) >= _mean(metrics['bm25']['SessR']) >= _mean(metrics['bm25']['R']) else 'FAIL'} ] BM25 Hit@k >= SessR@k >= R@k")
        print(f"  [ {'PASS' if _mean(metrics['sbert']['Hit']) >= _mean(metrics['sbert']['SessR']) >= _mean(metrics['sbert']['R']) else 'FAIL'} ] SBERT Hit@k >= SessR@k >= R@k")
        print(f"  [ {'PASS' if _mean(metrics['ppr']['Hit']) >= _mean(metrics['ppr']['SessR']) >= _mean(metrics['ppr']['R']) else 'FAIL'} ] PPR Hit@k >= SessR@k >= R@k")
        print(f"  [ {'PASS' if m_p >= m_s > m_b else 'FAIL'} ] SessR ordering: PPR >= SBERT > BM25")
        print(f"  [ {'PASS' if m_s > 0.4 else 'FAIL'} ] SBERT SessR@{k} > 0.4")

    print(f"[INFO] Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
