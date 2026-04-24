from __future__ import annotations

import argparse
import asyncio
import time

from grace.datasets.longmemeval import load_longmemeval_s
from grace.qa.judge import LLMJudge
from grace.qa.reader import QwenReader
from grace.retrievers.bm25 import BM25Retriever


def main() -> None:
    p = argparse.ArgumentParser(description="5-question QA smoke test.")
    p.add_argument("--data", default="data/raw/longmemeval/longmemeval_s.json")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--reader-backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--reader-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--judge-backend", choices=("siliconflow", "deepseek", "local_vllm"), default="siliconflow")
    args = p.parse_args()

    t0 = time.time()
    samples = [s for s in load_longmemeval_s(args.data) if s.turns and s.answer][: args.n]
    qs = [s.question for s in samples]
    ctxs = []
    rid_all = []
    for s in samples:
        texts = [t.text for t in s.turns]
        rr = BM25Retriever(texts).retrieve(s.question, k=args.k)
        rid = [i for i, _, _ in rr]
        ctx = [x for _, _, x in rr]
        rid_all.append(rid)
        ctxs.append(ctx)

    reader = QwenReader(model_name=args.reader_model, backend=args.reader_backend)
    preds = reader.answer_batch(qs, ctxs)
    judge = LLMJudge(backend=args.judge_backend)
    judge_inputs = [
        {
            "question": s.question,
            "gold": s.answer or "",
            "predicted": ptext,
            "question_type": s.question_type,
        }
        for s, ptext in zip(samples, preds)
    ]
    judged = asyncio.run(judge.judge_batch(judge_inputs, concurrency=5))

    for i, (s, pred, jd) in enumerate(zip(samples, preds, judged)):
        print(f"\n[{i}] type={s.question_type}")
        print(f"Q: {s.question}")
        print(f"Gold: {s.answer}")
        print(f"Pred: {pred}")
        print(f"Correct: {jd.get('correct', False)}")
        print(f"Reason: {jd.get('reasoning', '')}")

    print(f"\nTotal wall-clock: {time.time() - t0:.2f}s")
    print(f"Judge stats: {judge.stats()}")


if __name__ == "__main__":
    main()
