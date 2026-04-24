# GRACE (Graph Retrieval And Conversational Evidence)

Codebase for the course project: **graph memory + retrieval baselines (BM25, SBERT, PPR) + query-conditioned GNN retriever** on **LongMemEval** / **LoCoMo-10**.

## Environment (conda)

### Why it feels slow / no progress

- **`conda env create -f environment.yml`** often sits a long time on **“Solving environment”** or **“Installing pip dependencies”** with little output. That is normal, not necessarily a bad mirror.
- Your machine already uses **Tsinghua mirrors** for conda defaults / conda-forge cloud and **Tuna PyPI** (`pip config`), so **slow installs are usually solver + large wheels** (torch, transformers, sentence-transformers), not “forgot to换源”.
- For **visible steps**, prefer the script below (or run `conda env create -f environment.yml -vv`).

### Recommended (step-by-step, more logs)

```bash
cd grace
chmod +x scripts/setup_grace_env.sh
bash scripts/setup_grace_env.sh
conda activate grace
```

### Alternative (single file)

```bash
cd grace
conda env create -f environment.yml -vv
conda activate grace
bash scripts/install_torch_pyg.sh
pip install -e .
```

PyTorch + PyG wheels must match your CUDA driver — edit `PYTORCH_INDEX` / `PYG_URL` in `scripts/install_torch_pyg.sh` for cu118 / CPU if cu121 wheels fail.

Verify:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"
python -c "import torch_geometric; print('pyg', torch_geometric.__version__)"
```

## Data

1. **LongMemEval (cleaned)** — Hugging Face `xiaowu0162/longmemeval-cleaned` → under `data/raw/longmemeval/`. The repo file is usually **`longmemeval_s_cleaned.json`** (~280MB). Scripts default to **`longmemeval_s.json`** — after download run:  
   `cd data/raw/longmemeval && ln -sf longmemeval_s_cleaned.json longmemeval_s.json`
2. **LoCoMo-10** — copy `locomo10.json` from `third_party/locomo/data/` to `data/raw/locomo10.json`.

```bash
# HF download (can look "stuck" for minutes: little stdout until files appear; check disk with `du -sh data/raw/longmemeval`)
conda activate grace
cd grace
HF_HUB_DISABLE_PROGRESS_BARS=0 huggingface-cli download xiaowu0162/longmemeval-cleaned \
  --repo-type dataset --local-dir data/raw/longmemeval/
ln -sf longmemeval_s_cleaned.json data/raw/longmemeval/longmemeval_s.json
```

Check:

```bash
python scripts/verify_data.py
```

## Third-party code (for HippoRAG / paper baselines)

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/OSU-NLP-Group/HippoRAG.git
git clone https://github.com/xiaowu0162/LongMemEval.git
git clone https://github.com/snap-research/locomo.git
```

Then follow HippoRAG’s README to run a small `demo_local` (OpenIE + KG + PPR) with your own API key in `.env` (see `.env.example`).

## This repo: quick runs

| Script | Purpose |
|--------|--------|
| `bash scripts/run_smoke.sh` | Import torch/pyg; runs `test_bm25` if data exists |
| `python scripts/test_bm25.py` | BM25 on first LongMemEval question |
| `python scripts/eval_retrievers.py --limit 50` | Mean Recall@5 for BM25 / SBERT / PPR |
| `python scripts/train_gnn.py --data data/raw/longmemeval/longmemeval_s.json` | Train query-conditioned GNN |
| `python scripts/test_qa_smoke.py` | 5-question QA smoke (reader + judge) |
| `python scripts/run_full_eval.py --retriever sbert --limit 30 --k 5 --seed 42 --sbert-local-only --judge-backend siliconflow --output experiments/results/sbert_k5.json` | End-to-end retrieval + QA Accuracy |
| `bash scripts/run_all_experiments.sh` | verify + short eval |
| `python scripts/plot_pareto.py --input results.csv --out experiments/results/pareto.png` | Pareto-style plot |

QA reader defaults to `vLLM` with Qwen2.5-7B-Instruct. If `vllm` is unavailable, switch to transformers fallback by passing `--reader-backend transformers`.

## Layout

- `src/grace/` — package (`datasets`, `graphs`, `retrievers`, `models`, `eval`, `utils`)
- `configs/` — YAML; start from `configs/default.yaml`
- `data/raw/` — downloaded JSON (gitignored)
- `experiments/checkpoints`, `experiments/results` — outputs
- `third_party/` — cloned upstream repos (gitignored)

## Config matrix (stubs)

`configs/entity_kg_*.yaml` and `configs/sentence_graph_*.yaml` name each cell of the 2×4 design; `graph_kind: sentence_graph` is fully supported in code, `entity_kg` currently uses a sentence-graph-style stub until OpenIE/triples are wired.

## License

Project code: follow your course policy. Third-party repos keep their own licenses.
