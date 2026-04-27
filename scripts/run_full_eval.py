from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import string
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


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_text(text: str | None) -> str:
    text = str(text or "").lower()
    text = text.translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def _answer_in_context(answer: str | None, ctx: list[dict], max_answer_chars: int = 120) -> bool:
    ans = _normalize_text(answer)
    if not ans or len(ans) > max_answer_chars:
        return False
    ctx_text = _normalize_text(" ".join(str(t.get("content", "")) for t in ctx))
    return ans in ctx_text


def _lexical_answer_match(gold: str | None, pred: str | None, max_gold_words: int = 8) -> bool:
    g = _normalize_text(gold)
    p = _normalize_text(pred)
    if not g or not p:
        return False
    if len(g.split()) > max_gold_words:
        return False
    return g in p or p in g


def _is_idk(pred: str | None) -> bool:
    p = _normalize_text(pred)
    return p in {"i dont know", "i do not know", "dont know", "unknown"}


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


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _select_diverse_by_session(
    rows: list[tuple[int, float, str]],
    turn_to_session: dict[int, str],
    k: int,
    penalty: float,
) -> list[tuple[int, float, str]]:
    """Greedy session-diverse selection over already scored candidate turns."""
    selected: list[tuple[int, float, str]] = []
    selected_counts: dict[str, int] = {}
    remaining = list(rows)
    while remaining and len(selected) < k:
        best_pos = 0
        best_score = float("-inf")
        for pos, (gid, score, _) in enumerate(remaining):
            sid = str(turn_to_session.get(gid, "unknown"))
            adjusted = float(score) - penalty * selected_counts.get(sid, 0)
            if adjusted > best_score:
                best_score = adjusted
                best_pos = pos
        chosen = remaining.pop(best_pos)
        selected.append(chosen)
        sid = str(turn_to_session.get(chosen[0], "unknown"))
        selected_counts[sid] = selected_counts.get(sid, 0) + 1
    return selected


_PACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

_NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "hundred",
    "thousand",
}
_MONTH_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "today",
    "tomorrow",
    "yesterday",
    "week",
    "month",
    "year",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}
_COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "green",
    "grey",
    "gray",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
_LOCATION_HINTS = {
    "airport",
    "apartment",
    "bar",
    "beach",
    "cafe",
    "campus",
    "city",
    "clinic",
    "college",
    "garden",
    "gym",
    "hotel",
    "house",
    "library",
    "museum",
    "office",
    "park",
    "restaurant",
    "room",
    "school",
    "shop",
    "store",
    "street",
    "studio",
    "theater",
    "university",
}


def _pack_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {w for w in words if len(w) > 2 and w not in _PACK_STOPWORDS}


def _lexical_overlap_score(question: str, text: str) -> float:
    q_tokens = _pack_tokens(question)
    if not q_tokens:
        return 0.0
    t_tokens = _pack_tokens(text)
    return len(q_tokens & t_tokens) / len(q_tokens)


def _has_number(text: str) -> bool:
    low = str(text).lower()
    return bool(re.search(r"\d", low)) or any(w in low.split() for w in _NUMBER_WORDS)


def _has_time(text: str) -> bool:
    low = str(text).lower()
    return bool(re.search(r"\b\d{1,2}[:/.-]\d{1,2}\b|\b\d{4}\b", low)) or any(w in low for w in _MONTH_WORDS)


def _has_entity_like(text: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", str(text))) or '"' in str(text) or "'" in str(text)


def _answer_like_bonus(question: str, text: str) -> float:
    q = str(question).lower()
    t = str(text).lower()
    bonus = 0.0
    if any(x in q for x in ("how many", "how much", "number", "amount", "price", "cost", "total", "times")):
        bonus += 0.35 if _has_number(text) else -0.10
    if any(x in q for x in ("when", "what time", "what date", "how long", "day", "week", "month", "year")):
        bonus += 0.30 if _has_time(text) else -0.05
    if "where" in q or "which place" in q:
        bonus += 0.25 if (_has_entity_like(text) or any(w in t for w in _LOCATION_HINTS)) else -0.05
    if "color" in q or "colour" in q:
        bonus += 0.30 if any(w in t for w in _COLOR_WORDS) else -0.05
    if any(x in q for x in ("who", "which", "what is the name", "name of")):
        bonus += 0.15 if _has_entity_like(text) else 0.0
    return bonus


def _select_answer_aware_context(
    question: str,
    rows: list[tuple[int, float, str]],
    turn_to_session: dict[int, str],
    k: int,
    pack_weight: float,
    session_penalty: float,
) -> list[tuple[int, float, str]]:
    """Greedy final context packing tuned for answer-containing turns."""
    selected: list[tuple[int, float, str]] = []
    selected_counts: dict[str, int] = {}
    remaining = list(rows)
    while remaining and len(selected) < k:
        best_pos = 0
        best_score = float("-inf")
        for pos, (gid, score, text) in enumerate(remaining):
            sid = str(turn_to_session.get(gid, "unknown"))
            answer_score = 0.45 * _lexical_overlap_score(question, text) + _answer_like_bonus(question, text)
            adjusted = float(score) + pack_weight * answer_score - session_penalty * selected_counts.get(sid, 0)
            if adjusted > best_score:
                best_score = adjusted
                best_pos = pos
        chosen = remaining.pop(best_pos)
        selected.append(chosen)
        sid = str(turn_to_session.get(chosen[0], "unknown"))
        selected_counts[sid] = selected_counts.get(sid, 0) + 1
    return selected


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    return [p.strip() for p in parts if p.strip()]


def _compress_contexts_for_reader(
    question: str,
    ctx: list[dict],
    sentences_per_turn: int,
    max_chars: int,
) -> list[dict]:
    """Keep query-relevant sentences from retrieved turns to reduce reader noise."""
    compressed: list[dict] = []
    for turn in ctx:
        content = str(turn.get("content", ""))
        sentences = _split_sentences(content)
        if len(content) <= 360 or len(sentences) <= sentences_per_turn:
            new_content = content
        else:
            scored = []
            for idx, sent in enumerate(sentences):
                score = _lexical_overlap_score(question, sent) + _answer_like_bonus(question, sent)
                scored.append((idx, score, sent))
            picked = sorted(scored, key=lambda x: x[1], reverse=True)[:sentences_per_turn]
            picked = sorted(picked, key=lambda x: x[0])
            new_content = " ".join(sent for _, _, sent in picked)
        row = dict(turn)
        row["content"] = new_content.strip()
        compressed.append(row)

    if max_chars <= 0:
        return compressed
    total = 0
    trimmed: list[dict] = []
    for turn in compressed:
        row = dict(turn)
        content = str(row.get("content", ""))
        budget = max_chars - total
        if budget <= 0:
            break
        if len(content) > budget:
            content = content[: max(0, budget)].rsplit(" ", 1)[0].strip()
        if content:
            row["content"] = content
            trimmed.append(row)
            total += len(content)
    return trimmed


def _expand_contexts_session_window(
    sample: LongMemSample,
    anchor_ids: list[int],
    window_radius: int,
    max_turns: int,
    max_chars: int,
) -> list[dict]:
    """Expand retrieved anchor turns into local same-session windows."""
    tm = _build_turn_meta(sample)
    candidates: dict[int, tuple[int, int, int]] = {}
    n_turns = len(sample.turns)
    for anchor_rank, anchor_gid in enumerate(anchor_ids, start=1):
        anchor_sid = sample.turn_to_session.get(anchor_gid)
        if anchor_sid is None:
            continue
        lo = max(0, anchor_gid - window_radius)
        hi = min(n_turns - 1, anchor_gid + window_radius)
        for gid in range(lo, hi + 1):
            if sample.turn_to_session.get(gid) != anchor_sid:
                continue
            dist = abs(gid - anchor_gid)
            old = candidates.get(gid)
            # Prefer all original anchors first, then add nearby context.
            key = (dist, anchor_rank, anchor_gid)
            if old is None or key < old:
                candidates[gid] = key

    chosen = sorted(candidates.items(), key=lambda x: x[1])
    if max_turns > 0:
        chosen = chosen[:max_turns]
    chosen_ids = sorted(gid for gid, _ in chosen)
    candidate_meta = {gid: meta for gid, meta in candidates.items()}

    out: list[dict] = []
    total_chars = 0
    for gid in chosen_ids:
        dist, anchor_rank, anchor_gid = candidate_meta[gid]
        row = dict(
            tm.get(
                gid,
                {
                    "session_id": str(sample.turn_to_session.get(gid, "unknown")),
                    "session_date": "",
                    "role": "unknown",
                    "content": sample.turns[gid].text if 0 <= gid < len(sample.turns) else "",
                },
            )
        )
        content_len = len(str(row.get("content", "")))
        if max_chars > 0 and out and total_chars + content_len > max_chars:
            continue
        row["window_turn_id"] = gid
        row["window_anchor_turn_id"] = anchor_gid
        row["window_anchor_rank"] = anchor_rank
        row["window_distance"] = dist
        row["context_mode"] = "session_window"
        out.append(row)
        total_chars += content_len
    return out


def _retrieve_one(
    s: LongMemSample,
    retriever: str,
    k: int,
    graph_topk: int,
    sbert_model: str,
    graph_session_window: int = 1,
    graph_session_semantic_topk: int = 0,
    gnn_retriever: GNNRetriever | None = None,
    rerank_pool: int = 20,
    fusion_gnn_weight: float = 0.5,
    fusion_sbert_weight: float = 0.3,
    fusion_bm25_weight: float = 0.2,
    diversify_session_penalty: float = 0.0,
    answer_pack_weight: float = 0.35,
    answer_pack_session_penalty: float = 0.05,
) -> tuple[list[int], list[dict]]:
    texts = [t.text for t in s.turns]
    retrieval_meta: dict[int, dict] = {}
    if retriever == "bm25":
        r = BM25Retriever(texts).retrieve(s.question, k=k)
    elif retriever == "sbert":
        r = SBERTRetriever(texts, model_name=sbert_model).retrieve(s.question, k=k)
    elif retriever == "ppr":
        g = build_sentence_graph(
            s.turns,
            topk=graph_topk,
            sbert_model=sbert_model,
            session_window=graph_session_window,
            session_semantic_topk=graph_session_semantic_topk,
        )
        r = PPRRetriever(g, sbert_model=sbert_model).retrieve(s.question, k=k)
    elif retriever == "gnn":
        if gnn_retriever is None:
            raise ValueError("GNN retriever selected but no checkpoint-loaded retriever was provided.")
        g = build_sentence_graph(
            s.turns,
            topk=graph_topk,
            sbert_model=sbert_model,
            session_window=graph_session_window,
            session_semantic_topk=graph_session_semantic_topk,
        )
        r = gnn_retriever.retrieve(s.question, g.to_torch(device=gnn_retriever.device), k=k)
    elif retriever == "gnn+sbert_rerank":
        if gnn_retriever is None:
            raise ValueError("gnn+sbert_rerank selected but no checkpoint-loaded retriever was provided.")
        g = build_sentence_graph(
            s.turns,
            topk=graph_topk,
            sbert_model=sbert_model,
            session_window=graph_session_window,
            session_semantic_topk=graph_session_semantic_topk,
        )
        pool_k = max(k, rerank_pool)
        gnn_ranked = gnn_retriever.retrieve(
            s.question, g.to_torch(device=gnn_retriever.device), k=pool_k
        )
        cand_ids = [i for i, _, _ in gnn_ranked]
        cand_texts = [g.node_texts[i] for i in cand_ids]
        sb_ranked = SBERTRetriever(cand_texts, model_name=sbert_model).retrieve(s.question, k=k)
        r = [(cand_ids[i], score, text) for i, score, text in sb_ranked]
    elif retriever in ("gnn+fusion_rerank", "gnn+fusion_diverse_rerank", "gnn+fusion_answer_pack"):
        if gnn_retriever is None:
            raise ValueError(f"{retriever} selected but no checkpoint-loaded retriever was provided.")
        g = build_sentence_graph(
            s.turns,
            topk=graph_topk,
            sbert_model=sbert_model,
            session_window=graph_session_window,
            session_semantic_topk=graph_session_semantic_topk,
        )
        pool_k = max(k, rerank_pool)
        gnn_ranked = gnn_retriever.retrieve(
            s.question, g.to_torch(device=gnn_retriever.device), k=pool_k
        )
        cand_ids = [i for i, _, _ in gnn_ranked]
        cand_texts = [g.node_texts[i] for i in cand_ids]

        gnn_scores = _minmax([float(score) for _, score, _ in gnn_ranked])
        sbert_scores = [0.0 for _ in cand_ids]
        for local_i, score, _ in SBERTRetriever(cand_texts, model_name=sbert_model).retrieve(
            s.question, k=len(cand_ids)
        ):
            sbert_scores[local_i] = float(score)
        sbert_scores = _minmax(sbert_scores)

        bm25_scores = [0.0 for _ in cand_ids]
        for local_i, score, _ in BM25Retriever(cand_texts).retrieve(s.question, k=len(cand_ids)):
            bm25_scores[local_i] = float(score)
        bm25_scores = _minmax(bm25_scores)

        fused = []
        for local_i, gid in enumerate(cand_ids):
            score = (
                fusion_gnn_weight * gnn_scores[local_i]
                + fusion_sbert_weight * sbert_scores[local_i]
                + fusion_bm25_weight * bm25_scores[local_i]
            )
            retrieval_meta[gid] = {
                "retrieval_score": float(score),
                "gnn_score": float(gnn_scores[local_i]),
                "sbert_score": float(sbert_scores[local_i]),
                "bm25_score": float(bm25_scores[local_i]),
            }
            fused.append((gid, score, g.node_texts[gid]))
        ranked = sorted(fused, key=lambda x: x[1], reverse=True)
        if retriever == "gnn+fusion_diverse_rerank":
            r = _select_diverse_by_session(
                ranked,
                s.turn_to_session,
                k=k,
                penalty=diversify_session_penalty,
            )
        elif retriever == "gnn+fusion_answer_pack":
            r = _select_answer_aware_context(
                s.question,
                ranked,
                s.turn_to_session,
                k=k,
                pack_weight=answer_pack_weight,
                session_penalty=answer_pack_session_penalty,
            )
        else:
            r = ranked[:k]
    else:
        raise ValueError(f"Unsupported retriever: {retriever}")
    ids = [i for i, _, _ in r]
    tm = _build_turn_meta(s)
    ctx = []
    for rank, (i, score, _) in enumerate(r, start=1):
        row = dict(
            tm.get(
                i,
                {
                    "session_id": str(s.turn_to_session.get(i, "unknown")),
                    "session_date": "",
                    "role": "unknown",
                    "content": s.turns[i].text if 0 <= i < len(s.turns) else "",
                },
            )
        )
        row["evidence_rank"] = rank
        row["retriever"] = retriever
        row["retrieval_score"] = float(score)
        row.update(retrieval_meta.get(i, {}))
        ctx.append(row)
    return ids, ctx


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end retrieval + QA + judge evaluation.")
    p.add_argument("--data", default="data/raw/longmemeval/longmemeval_s.json")
    p.add_argument(
        "--retriever",
        choices=(
            "bm25",
            "sbert",
            "ppr",
            "gnn",
            "gnn+sbert_rerank",
            "gnn+fusion_rerank",
            "gnn+fusion_diverse_rerank",
            "gnn+fusion_answer_pack",
        ),
        default="sbert",
    )
    p.add_argument("--checkpoint", default="", help="Required when --retriever gnn")
    p.add_argument("--gnn-arch", choices=("sage", "sage_res", "sage_skip", "sage_qa", "gat"), default="sage")
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument("--rerank-pool", type=int, default=20)
    p.add_argument("--fusion-gnn-weight", type=float, default=0.5)
    p.add_argument("--fusion-sbert-weight", type=float, default=0.3)
    p.add_argument("--fusion-bm25-weight", type=float, default=0.2)
    p.add_argument(
        "--diversify-session-penalty",
        type=float,
        default=0.15,
        help="Score penalty for selecting additional turns from an already selected session.",
    )
    p.add_argument(
        "--answer-pack-weight",
        type=float,
        default=0.35,
        help="Weight for answer-like lexical/type cues in gnn+fusion_answer_pack.",
    )
    p.add_argument(
        "--answer-pack-session-penalty",
        type=float,
        default=0.05,
        help="Small session repeat penalty used by gnn+fusion_answer_pack.",
    )
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graph-topk", type=int, default=5)
    p.add_argument(
        "--graph-session-window",
        type=int,
        default=1,
        help="Connect turns within this same-session distance; 1 keeps the legacy chain graph.",
    )
    p.add_argument(
        "--graph-session-semantic-topk",
        type=int,
        default=0,
        help="Add this many same-session semantic neighbors per node; 0 keeps legacy graph.",
    )
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sbert-local-only", action="store_true")
    p.add_argument("--reader-backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--reader-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--reader-batch-size", type=int, default=16)
    p.add_argument(
        "--reader-prompt-mode",
        choices=("plain", "ranked", "retrieval_aware"),
        default="plain",
        help="Prompt format used by the reader.",
    )
    p.add_argument(
        "--context-mode",
        choices=("full", "sentence_pack", "session_window"),
        default="full",
        help="Optional reader-side context compression after retrieval.",
    )
    p.add_argument("--context-sentences-per-turn", type=int, default=2)
    p.add_argument("--context-max-chars", type=int, default=1400)
    p.add_argument("--window-radius", type=int, default=1)
    p.add_argument("--window-max-turns", type=int, default=12)
    p.add_argument("--window-max-chars", type=int, default=9000)
    p.add_argument("--judge-backend", choices=("siliconflow", "deepseek", "local_vllm"), default="siliconflow")
    p.add_argument("--judge-concurrency", type=int, default=10)
    p.add_argument("--judge-cache-dir", default="data/processed/judge_cache")
    p.add_argument("--judge-max-retries", type=int, default=3)
    p.add_argument(
        "--no-save-contexts",
        action="store_true",
        help="Do not store retrieved context text in per-question audit output.",
    )
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
    if args.retriever in (
        "gnn",
        "gnn+sbert_rerank",
        "gnn+fusion_rerank",
        "gnn+fusion_diverse_rerank",
        "gnn+fusion_answer_pack",
    ):
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
                    self.model = QueryGNN(
                        in_dim=384,
                        hidden_dim=256,
                        num_layers=2,
                        query_dim=384,
                        arch=args.gnn_arch,
                        gat_heads=args.gat_heads,
                    ).to(self.device)
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
                arch=args.gnn_arch,
                gat_heads=args.gat_heads,
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
            graph_session_window=args.graph_session_window,
            graph_session_semantic_topk=args.graph_session_semantic_topk,
            gnn_retriever=gnn_retriever,
            rerank_pool=args.rerank_pool,
            fusion_gnn_weight=args.fusion_gnn_weight,
            fusion_sbert_weight=args.fusion_sbert_weight,
            fusion_bm25_weight=args.fusion_bm25_weight,
            diversify_session_penalty=args.diversify_session_penalty,
            answer_pack_weight=args.answer_pack_weight,
            answer_pack_session_penalty=args.answer_pack_session_penalty,
        )
        retrieved_ids.append(ids)
        contexts.append(ctx)
    if args.context_mode == "sentence_pack":
        contexts = [
            _compress_contexts_for_reader(
                s.question,
                ctx,
                sentences_per_turn=args.context_sentences_per_turn,
                max_chars=args.context_max_chars,
            )
            for s, ctx in zip(subset, contexts)
        ]
    elif args.context_mode == "session_window":
        contexts = [
            _expand_contexts_session_window(
                s,
                ids,
                window_radius=args.window_radius,
                max_turns=args.window_max_turns,
                max_chars=args.window_max_chars,
            )
            for s, ids in zip(subset, retrieved_ids)
        ]
    t_retr = time.time() - t_retr_start

    t_reader_start = time.time()
    reader = QwenReader(
        model_name=args.reader_model,
        backend=args.reader_backend,
        prompt_mode=args.reader_prompt_mode,
    )
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
    all_answer_ctx: list[float] = []
    all_idk: list[float] = []
    all_empty: list[float] = []
    all_judge_err: list[float] = []
    all_lexical_match: list[float] = []

    for s, rid, ctx, pred, jo in zip(subset, retrieved_ids, contexts, preds, judge_out):
        pos = set(s.evidence_global_ids)
        sr = session_recall_at_k(rid, s.turn_to_session, s.gold_sessions, args.k)
        rr = recall_at_k(rid, pos, args.k)
        hh = hit_at_k(rid, pos, args.k)
        mr = mrr(rid, pos)
        correct = bool(jo.get("correct", False))
        answer_ctx = _answer_in_context(s.answer, ctx)
        idk = _is_idk(pred)
        empty = not str(pred or "").strip()
        judge_err = "judge_error" in str(jo.get("reasoning", ""))
        lexical_match = _lexical_answer_match(s.answer, pred)
        all_acc.append(1.0 if correct else 0.0)
        all_r.append(rr)
        all_h.append(hh)
        all_sr.append(sr)
        all_mrr.append(mr)
        all_answer_ctx.append(1.0 if answer_ctx else 0.0)
        all_idk.append(1.0 if idk else 0.0)
        all_empty.append(1.0 if empty else 0.0)
        all_judge_err.append(1.0 if judge_err else 0.0)
        all_lexical_match.append(1.0 if lexical_match else 0.0)
        by_type_acc[s.question_type].append(1.0 if correct else 0.0)
        by_type_sessr[s.question_type].append(sr)
        row = {
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
            "recall_at_5": rr,
            "hit_at_5": hh,
            "mrr": mr,
            "answer_in_context_at_5": answer_ctx,
            "idk_prediction": idk,
            "empty_prediction": empty,
            "lexical_answer_match": lexical_match,
            "judge_cached": bool(jo.get("cached", False)),
            "judge_reasoning": jo.get("reasoning", ""),
            "judge_raw_response": jo.get("raw_response", ""),
        }
        if not args.no_save_contexts:
            row["retrieved_contexts"] = ctx
        per_question.append(row)

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
            "graph_topk": args.graph_topk,
            "graph_session_window": args.graph_session_window,
            "graph_session_semantic_topk": args.graph_session_semantic_topk,
            "reader_backend": args.reader_backend,
            "reader_model": args.reader_model,
            "reader_prompt_mode": args.reader_prompt_mode,
            "judge_backend": args.judge_backend,
            "answer_pack_weight": args.answer_pack_weight,
            "answer_pack_session_penalty": args.answer_pack_session_penalty,
            "context_mode": args.context_mode,
            "context_sentences_per_turn": args.context_sentences_per_turn,
            "context_max_chars": args.context_max_chars,
            "window_radius": args.window_radius,
            "window_max_turns": args.window_max_turns,
            "window_max_chars": args.window_max_chars,
        },
        "overall": {
            "accuracy": _mean(all_acc),
            "sess_recall_at_5": _mean(all_sr),
            "recall_at_5": _mean(all_r),
            "hit_at_5": _mean(all_h),
            "mrr": _mean(all_mrr),
            "answer_in_context_at_5": _mean(all_answer_ctx),
            "idk_rate": _mean(all_idk),
            "empty_prediction_rate": _mean(all_empty),
            "judge_error_rate": _mean(all_judge_err),
            "lexical_answer_match_rate": _mean(all_lexical_match),
            "n": len(subset),
        },
        "by_type": by_type,
        "per_question": per_question,
        "stats": {
            **judge.stats(),
            "avg_predicted_words": _mean([len(str(p).split()) for p in preds]),
            "audit_note": (
                "answer_in_context_at_5 uses normalized exact substring matching "
                "for gold answers up to 120 characters; long/free-form answers are counted false."
            ),
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
    print(
        f"audit answer_ctx@5={out_obj['overall']['answer_in_context_at_5']:.3f} "
        f"idk={out_obj['overall']['idk_rate']:.3f} "
        f"judge_err={out_obj['overall']['judge_error_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
