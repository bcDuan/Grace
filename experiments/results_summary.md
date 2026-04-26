# GRACE Experiment Results Summary

Date: 2026-04-26

This file is a lightweight, git-tracked summary of the main experiment results. Full JSON outputs and model checkpoints are intentionally not committed because `experiments/results/**` and `*.pt` are ignored.

## Main Result

Evaluation setting: LongMemEval-S, stratified n=100, top-k=5, Qwen2.5-7B reader, LLM judge.

| Method | QA Acc | SessR@5 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.400 | 0.744 | 0.850 | 0.783 |
| SBERT | 0.400 | 0.799 | 0.930 | 0.869 |
| PPR | 0.350 | 0.819 | 0.930 | 0.870 |
| Original GNN | 0.230 | 0.832 | 0.930 | 0.878 |
| Original GNN + SBERT rerank | 0.320 | 0.850 | 0.960 | 0.907 |
| Answer-Aware GNN v2 + SBERT rerank | 0.340 | 0.891 | 0.980 | 0.940 |
| Pairwise Graph GNN | 0.350 | 0.875 | 0.960 | 0.946 |
| Pairwise Graph GNN + SBERT rerank | 0.360 | 0.890 | 0.990 | 0.950 |
| Pairwise Graph GNN + Fusion rerank | **0.430** | **0.913** | **0.970** | **0.950** |

## Key Takeaways

- The original trained GNN had strong session localization but low QA accuracy: Acc 0.230, SessR@5 0.832.
- Answer-aware supervision improved the GNN by distinguishing answer-bearing turns from other turns in evidence sessions.
- Pairwise graph ranking further improved answer-bearing turn ranking.
- Graph-guided fusion reranking achieved the best result: Acc 0.430, SessR@5 0.913.
- Compared with BM25/SBERT, the final method improves QA accuracy from 0.400 to 0.430 while keeping much stronger session recall.

## Final Method

Recommended method name for presentation/report:

```text
Answer-Aware Pairwise Graph Retriever + Graph-Guided Fusion Reranker
```

Fusion score:

```text
final_score = 0.5 * GNN_score + 0.3 * SBERT_score + 0.2 * BM25_score
```

Interpretation:

- GNN provides graph-based memory localization and candidate scoring.
- SBERT provides local semantic matching.
- BM25 provides lexical/numeric/time phrase matching.
- The final pipeline keeps graph learning as the core retrieval mechanism while addressing answer-level evidence selection.

## Important Saved Artifacts On Remote Server

Project root:

```text
/hpc2hdd/home/bduan436/Work_space/GraphMem/grace
```

Best checkpoint:

```text
experiments/checkpoints/pairwise_ft_epoch8.pt
```

Best full result JSON:

```text
experiments/results/pairwise_ft_ep8_fusion_n100.json
```

Best run log:

```text
experiments/results/pairwise_ft_ep8_fusion_n100.log
```

Failure cases:

```text
experiments/results/pairwise_ft_fusion_failure_cases.md
```

Iteration notes:

```text
/hpc2hdd/home/bduan436/Work_space/GraphMem/工作空间/迭代方案/GRACE_迭代方案_2026-04-26.md
```

## Reproduction Commands

Pairwise fine-tuning from the answer-aware checkpoint:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_gnn.py \
  --data data/raw/longmemeval/longmemeval_s.json \
  --epochs 6 \
  --skip_qa_eval \
  --batch_size 4 \
  --lr 5e-4 \
  --resume_checkpoint experiments/checkpoints/answer_aware_v2_epoch9.pt \
  --rank_lam 0.3 \
  --rank_margin 0.5 \
  --hard_negatives 16 \
  --checkpoint_pattern experiments/checkpoints/pairwise_ft_epoch{ep}.pt \
  --curve_out experiments/results/pairwise_ft_retrieval_curve.json
```

Final n100 fusion evaluation:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_full_eval.py \
  --retriever gnn+fusion_rerank \
  --checkpoint experiments/checkpoints/pairwise_ft_epoch8.pt \
  --rerank-pool 20 \
  --fusion-gnn-weight 0.5 \
  --fusion-sbert-weight 0.3 \
  --fusion-bm25-weight 0.2 \
  --limit 100 \
  --k 5 \
  --seed 42 \
  --sbert-local-only \
  --reader-backend transformers \
  --reader-batch-size 2 \
  --judge-backend siliconflow \
  --judge-cache-dir data/processed/judge_cache \
  --output experiments/results/pairwise_ft_ep8_fusion_n100.json
```

## Ablation: Session Diversification

We also tested a graph-guided session diversification variant. After fusion reranking, it greedily selects turns while applying a small penalty to additional turns from already selected sessions. The goal is to improve multi-session coverage.

| Method | QA Acc | SessR@5 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Fusion rerank | 0.430 | 0.913 | 0.970 | 0.950 |
| Fusion + session diversification, penalty=0.05 | 0.420 | 0.933 | 0.970 | 0.953 |

Conclusion:

- Session diversification improves session coverage: SessR@5 0.913 -> 0.933.
- It slightly reduces QA accuracy: 0.430 -> 0.420.
- Therefore, the non-diverse fusion reranker remains the main method, while diversification is useful as an ablation showing the tradeoff between broader graph coverage and answer-level precision.

Saved ablation result:

```text
experiments/results/pairwise_ft_ep8_fusion_diverse_p005_n100.json
```
