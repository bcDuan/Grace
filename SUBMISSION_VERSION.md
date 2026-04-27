# Course Project Submission Version

Date: 2026-04-27

This commit marks the code version intended for the graph learning course project submission.

## Main Method

Pairwise GraphSAGE + Graph-Guided Fusion Rerank + Qwen2.5-14B Reader, k=10.

The graph retriever uses conversation turns as nodes and combines semantic edges with same-session chain edges. The final retriever uses the trained GNN score together with SBERT and BM25 scores for graph-guided evidence reranking.

## Main Result

Validation split: 100 LongMemEval-S questions.

```text
Accuracy: 0.550
SessR@10: 0.949
Recall@10: 0.546
Hit@10: 0.970
MRR: 0.950
IDK rate: 0.160
```

Main result file:

```text
experiments/results/pairwise_ft_ep8_fusion_qwen14b_k10_n100.json
```

Main checkpoint:

```text
experiments/checkpoints/pairwise_ft_epoch8.pt
```

## Reproduction Notes

Create the environment:

```bash
conda env create -f environment.yml
conda activate grace
pip install -e .
```

Run the main evaluation:

```bash
python scripts/run_full_eval.py \
  --data data/raw/longmemeval/longmemeval_s.json \
  --retriever gnn+fusion_rerank \
  --checkpoint experiments/checkpoints/pairwise_ft_epoch8.pt \
  --gnn-arch sage \
  --limit 100 \
  --k 10 \
  --seed 42 \
  --graph-topk 5 \
  --graph-session-window 1 \
  --graph-session-semantic-topk 0 \
  --rerank-pool 20 \
  --fusion-gnn-weight 0.5 \
  --fusion-sbert-weight 0.3 \
  --fusion-bm25-weight 0.2 \
  --reader-backend transformers \
  --reader-model Qwen/Qwen2.5-14B-Instruct \
  --reader-batch-size 1 \
  --reader-prompt-mode plain \
  --judge-backend siliconflow \
  --judge-cache-dir data/processed/judge_cache \
  --output experiments/results/pairwise_ft_ep8_fusion_qwen14b_k10_n100.json
```

## Additional Ablations

Important ablation files from this version:

```text
experiments/results/pairwise_ft_ep8_fusion_qwen14b_n100.json
experiments/results/pairwise_ft_ep8_fusion_qwen14b_k10_n30.json
experiments/results/session_graph_w2_ep3_fusion_qwen14b_k10_n100.json
experiments/results/pairwise_ft_retrieval_k10_curve.json
experiments/results/session_graph_w2_retrieval_k10_curve.json
```

The session-window graph variant improved coverage metrics but did not improve final QA accuracy, so the submission version keeps the legacy graph construction.
