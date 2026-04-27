from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class QwenReader:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        backend: str = "vllm",
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        cache_dir: str = "data/processed/reader_cache",
        prompt_mode: str = "plain",
    ):
        if prompt_mode not in {"plain", "ranked", "retrieval_aware"}:
            raise ValueError(f"Unknown reader prompt_mode: {prompt_mode}")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        self.prompt_mode = prompt_mode
        self.max_new_tokens = min(max_new_tokens, 64)
        self.temperature = temperature
        self.stop_strings = ["\n\n", "Question:", "Past conversation"]

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if backend == "vllm":
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
            )
            self.sp = SamplingParams(
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                stop=self.stop_strings,
            )
        elif backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.model.eval()
            self.torch = torch
        else:
            raise ValueError(f"Unknown reader backend: {backend}")

    def _cache_key(self, question: str, context: list[dict[str, Any]]) -> str:
        text = json.dumps(
            {
                "q": question,
                "ctx": context,
                "m": self.model_name,
                "b": self._backend,
                "prompt_mode": self.prompt_mode,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _retrieval_header(self, turn: dict[str, Any]) -> str:
        fields = []
        if "evidence_rank" in turn:
            fields.append(f"rank={turn['evidence_rank']}")
        if self.prompt_mode == "ranked":
            sid = str(turn.get("session_id", "unknown"))
            sdate = str(turn.get("session_date", "") or "")
            fields = []
            if "evidence_rank" in turn:
                fields.append(f"rank={turn['evidence_rank']}")
            fields.append(f"session={sdate or sid}")
            fields.append(f"role={str(turn.get('role', 'unknown')).lower()}")
            return "[Evidence " + " | ".join(fields) + "]"
        if "retrieval_score" in turn:
            fields.append(f"score={float(turn['retrieval_score']):.3f}")
        if "gnn_score" in turn:
            fields.append(f"gnn={float(turn['gnn_score']):.3f}")
        if "sbert_score" in turn:
            fields.append(f"sbert={float(turn['sbert_score']):.3f}")
        if "bm25_score" in turn:
            fields.append(f"bm25={float(turn['bm25_score']):.3f}")
        sid = str(turn.get("session_id", "unknown"))
        sdate = str(turn.get("session_date", "") or "")
        fields.append(f"session={sdate or sid}")
        fields.append(f"role={str(turn.get('role', 'unknown')).lower()}")
        return "[Evidence " + " | ".join(fields) + "]"

    def _format_context(self, retrieved_turns: list[dict[str, Any]]) -> str:
        if self.prompt_mode in {"ranked", "retrieval_aware"}:
            blocks = []
            for t in retrieved_turns:
                role = str(t.get("role", "unknown")).upper()
                content = str(t.get("content", ""))
                blocks.append(f"{self._retrieval_header(t)}\n{role}: {content}")
            return "\n\n".join(blocks)

        by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in retrieved_turns:
            by_session[str(t.get("session_id", "unknown"))].append(t)

        blocks: list[str] = []
        session_order = sorted(
            by_session.keys(),
            key=lambda sid: str(by_session[sid][0].get("session_date", "")),
        )
        for sid in session_order:
            turns = by_session[sid]
            date_str = str(turns[0].get("session_date", "") or "")
            header = f"[Session {date_str}]" if date_str else f"[Session {sid}]"
            body = "\n".join(
                f"{str(t.get('role', 'unknown')).upper()}: {str(t.get('content', ''))}"
                for t in turns
            )
            blocks.append(f"{header}\n{body}")
        return "\n\n".join(blocks)

    def _build_messages(self, question: str, context_text: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a memory-augmented assistant. You will be given excerpts "
                    "from a user's past conversations and a question. Answer based ONLY "
                    "on the excerpts. Be concise: output the shortest answer that fully "
                    "answers the question. If the answer is not present in the excerpts, "
                    "output exactly: I don't know."
                    + (
                        " The excerpts are ordered by retrieval relevance. Prefer earlier "
                        "evidence when multiple excerpts conflict, but only answer when "
                        "the text itself supports the answer."
                        if self.prompt_mode == "ranked"
                        else ""
                    )
                    + (
                        " Each excerpt may include retrieval metadata such as rank, GNN, "
                        "SBERT, and BM25 scores. Treat these scores as hints: prefer "
                        "higher-ranked evidence when excerpts conflict, but only answer "
                        "when the text itself supports the answer."
                        if self.prompt_mode == "retrieval_aware"
                        else ""
                    )
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Past conversation excerpts:\n\n{context_text}\n\n"
                    f"Question: {question}\n\nShort answer:"
                ),
            },
        ]

    def _build_prompt(self, question: str, context_text: str) -> str:
        messages = self._build_messages(question, context_text)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _post_process_answer(self, text: str) -> str:
        t = text.strip()
        for stop in self.stop_strings:
            if stop in t:
                t = t.split(stop)[0]
        return t.strip()

    def answer_batch(
        self, questions: list[str], contexts: list[list[dict[str, Any]]]
    ) -> list[str]:
        prompts: list[str] = []
        cache_paths: list[Path] = []
        out: list[str | None] = [None] * len(questions)

        for i, (q, c) in enumerate(zip(questions, contexts)):
            cp = self.cache_dir / f"{self._cache_key(q, c)}.json"
            cache_paths.append(cp)
            if cp.exists():
                out[i] = json.loads(cp.read_text(encoding="utf-8"))["answer"]
            else:
                context_text = self._format_context(c)
                prompts.append(self._build_prompt(q, context_text))

        uncached_idx = [i for i, x in enumerate(out) if x is None]
        if uncached_idx:
            if self._backend == "vllm":
                outputs = self.llm.generate(prompts, self.sp)
                new_answers = [self._post_process_answer(o.outputs[0].text) for o in outputs]
            else:
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                ).to(self.model.device)
                with self.torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        temperature=None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                gen_ids = outputs[:, inputs["input_ids"].shape[1] :]
                texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                new_answers = [self._post_process_answer(t) for t in texts]

            for i, ans in zip(uncached_idx, new_answers):
                out[i] = ans
                cache_paths[i].write_text(
                    json.dumps({"answer": ans}, ensure_ascii=False), encoding="utf-8"
                )

        return [x or "" for x in out]
