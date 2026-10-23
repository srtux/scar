# SCAR Experimental Evaluation Summary

Evaluated on public $\tau^2$-bench airline dataset (400 simulations across Claude 3.7 Sonnet and GPT-4.1).

## Table 1: Failure Detection & Silent Failure Rates

| Metric | Measured Result |
| :--- | :---: |
| Total Simulations Evaluated | 400 |
| Total Failures | 188 |
| Failures with ZERO Tool Errors (Silent) | **171 / 188 (91.0%)** |
| Tool Error Code Alone Recall | 9.0% |
| User Pushback Alone Recall | **62.2%** |
| SCAR Dual Detector Recall | **65.4%** |

## Table 2: Memory Retention at 25% Token Budget

| Memory System | Fact Recall | Procedural Rule Recall | Overall Retention |
| :--- | :---: | :---: | :---: |
| Store-All FIFO (Truncation) | 50.0% | 100.0% | 75.0% |
| Atomic Facts (Mem0) | 100.0% | 35.0% | 67.5% |
| Tool-Outcome Only | 20.0% | 14.3% | 17.1% |
| **SCAR (Dual Key + Playbook)** | **100.0%** | **100.0%** | **100.0%** |

## Table 3: Repeat Mistake Reduction in Follow-up Trials

| Memory System | Evaluation Trials | Mistakes Repeated | Repeat Error Rate |
| :--- | :---: | :---: | :---: |
| Zero-Shot (No Memory) | 48 | 20 | 41.7% |
| Raw History (FIFO) | 48 | 20 | 41.7% |
| Atomic Facts (Mem0) | 48 | 20 | 41.7% |
| **SCAR (Contrastive Rule)** | **48** | **0** | **0.0%** |

**Absolute Error Reduction vs. Mem0:** **41.7%**
