from __future__ import annotations

import os

from openai import OpenAI

MODEL_MAP = {
    "siliconflow": {
        "fast": "Qwen/Qwen2.5-7B-Instruct",
        "strong": "Qwen/Qwen2.5-72B-Instruct",
    },
    "deepseek": {
        "fast": "deepseek-chat",
        "strong": "deepseek-chat",
    },
    "local_vllm": {
        "fast": "Qwen/Qwen2.5-7B-Instruct",
        "strong": "Qwen/Qwen2.5-7B-Instruct",
    },
}


def get_llm_client(backend: str = "siliconflow") -> OpenAI:
    """Unified OpenAI-compatible client factory."""
    if backend == "siliconflow":
        key = os.getenv("SILICONFLOW_API_KEY")
        if not key:
            raise ValueError("Missing SILICONFLOW_API_KEY for siliconflow backend.")
        return OpenAI(api_key=key, base_url="https://api.siliconflow.cn/v1")
    if backend == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("Missing DEEPSEEK_API_KEY for deepseek backend.")
        return OpenAI(api_key=key, base_url="https://api.deepseek.com")
    if backend == "local_vllm":
        return OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    raise ValueError(f"Unknown backend: {backend}")
