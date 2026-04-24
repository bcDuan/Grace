from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from grace.utils.llm_client import MODEL_MAP, get_llm_client

PRICE_PER_1M_CNY = {
    "siliconflow": {"fast": 1.0, "strong": 4.0},
    "deepseek": {"fast": 2.0, "strong": 2.0},
    "local_vllm": {"fast": 0.0, "strong": 0.0},
}


class LLMJudge:
    def __init__(
        self,
        backend: str = "siliconflow",
        model_tier: str = "strong",
        cache_dir: str = "data/processed/judge_cache",
        max_retries: int = 3,
    ):
        self.backend = backend
        self.client = get_llm_client(backend)
        self.model = MODEL_MAP[backend][model_tier]
        self.model_tier = model_tier
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.total_tokens_used = 0
        self.total_calls = 0
        self.cache_hits = 0

    def _cache_key(self, question: str, gold: str, predicted: str, question_type: str) -> str:
        key = f"{question}|{gold}|{predicted}|{question_type}|{self.model}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _build_prompt(self, question: str, gold: str, predicted: str, qtype: str) -> str:
        return (
            "You are an expert evaluator. Judge whether the predicted answer is "
            "semantically equivalent to the gold answer for the given question.\n\n"
            f"Question: {question}\n"
            f"Question type: {qtype}\n"
            f"Gold answer: {gold}\n"
            f"Predicted answer: {predicted}\n\n"
            "Respond in JSON with this schema only:\n"
            '{"correct": true/false, "reasoning": "brief explanation"}'
        )

    def _parse_response(self, text: str) -> dict[str, Any]:
        text = text.strip()
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        candidates = blocks if blocks else [text]
        for cand in candidates:
            try:
                obj = json.loads(cand)
                return {
                    "correct": bool(obj.get("correct", False)),
                    "reasoning": str(obj.get("reasoning", "")).strip(),
                }
            except Exception:
                pass
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return {
                    "correct": bool(obj.get("correct", False)),
                    "reasoning": str(obj.get("reasoning", "")).strip(),
                }
            except Exception:
                pass
        low = text.lower()
        return {
            "correct": ("true" in low or "correct" in low),
            "reasoning": text[:300],
        }

    def _estimate_cost_cny(self) -> float:
        price = PRICE_PER_1M_CNY[self.backend][self.model_tier]
        return (self.total_tokens_used / 1_000_000.0) * price

    def judge(self, question: str, gold: str, predicted: str, question_type: str) -> dict[str, Any]:
        cache_path = self.cache_dir / f"{self._cache_key(question, gold, predicted, question_type)}.json"
        if cache_path.exists():
            self.cache_hits += 1
            return {**json.loads(cache_path.read_text(encoding="utf-8")), "cached": True}

        prompt = self._build_prompt(question, gold, predicted, question_type)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                self.total_calls += 1
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.total_tokens_used += int(
                        getattr(usage, "total_tokens", 0)
                        or (getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0))
                    )
                result = self._parse_response(resp.choices[0].message.content or "")
                cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                return {**result, "cached": False}
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(2**attempt)
        return {"correct": False, "reasoning": f"judge_error: {last_err}", "cached": False}

    async def judge_batch(self, samples: list[dict[str, str]], concurrency: int = 10) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(s: dict[str, str]) -> dict[str, Any]:
            async with sem:
                return await asyncio.to_thread(
                    self.judge,
                    s["question"],
                    s.get("gold", ""),
                    s.get("predicted", ""),
                    s.get("question_type", "unknown"),
                )

        outs = await asyncio.gather(*[_one(s) for s in samples])
        return outs

    def stats(self) -> dict[str, Any]:
        total = self.total_calls + self.cache_hits
        return {
            "judge_cache_hit_rate": (self.cache_hits / total) if total else 0.0,
            "total_judge_calls": self.total_calls,
            "tokens_used": self.total_tokens_used,
            "est_cost_cny": round(self._estimate_cost_cny(), 6),
        }
