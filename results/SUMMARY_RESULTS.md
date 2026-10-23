# Experimental Summary

Every table below comes from live model and embedding runs with oracle fields stripped before detection and memory consolidation.

- **Model Backend:** `gemini-3.8-flash` (auxiliary: `gemini-3.5-flash-lite`, embeddings: `gemini-embedding-001`)
- **Runtime Telemetry:** 762 live LLM calls (+5,567 cached), 175 batch embedding calls (+4,052 cached), 4,052,871 prompt tokens, 75,840 completion tokens, 565s wall-clock.

## Failure Detection

Across 1,312 multi-turn simulations in tau2-bench (Airline and Retail), **328 of 403 failures (81.4%)** completed with zero tool execution errors (91.0% on Airline and 73.0% on Retail). Conversely, **125** passing sessions hit mid-session tool errors and recovered (95 separated from unresolved failures by SCAR).

Across 275 human-rated OpenHands coding sessions, 33 of 111 negative sessions (29.7%) had zero tool errors, while 93 positive sessions recovered from mid-session command errors.

### Benchmark Simulations

Evaluated on 1,312 tau2-bench sessions (403 unresolved failures and 909 resolved passes).

| Detector | TP | FP | TN | FN | Precision | Recall | FPR | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tool-Error Only | 75 | 125 | 784 | 328 | 37.5% | 18.6% | 13.8% | 24.9% |
| Naive Regex Pushback | 300 | 716 | 193 | 103 | 29.5% | 74.4% | 78.8% | 42.3% |
| Static Sentiment | 164 | 319 | 590 | 239 | 34.0% | 40.7% | 35.1% | 37.0% |
| Contextual User Pushback Only | 32 | 10 | 899 | 371 | 76.2% | 7.9% | 1.1% | 14.4% |
| **SCAR Transition Detector** | 220 | 122 | 787 | 183 | 64.3% | 54.6% | 13.4% | 59.1% |

### Human Coding Sessions

Evaluated on 275 OpenHands sessions (111 negative and 164 positive human ratings).

| Detector | TP | FP | TN | FN | Precision | Recall | FPR | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tool-Error Only | 78 | 93 | 71 | 33 | 45.6% | 70.3% | 56.7% | 55.3% |
| Naive Regex Pushback | 17 | 32 | 132 | 94 | 34.7% | 15.3% | 19.5% | 21.2% |
| Static Sentiment | 8 | 9 | 155 | 103 | 47.1% | 7.2% | 5.5% | 12.5% |
| Contextual User Pushback Only | 30 | 39 | 125 | 81 | 43.5% | 27.0% | 23.8% | 33.3% |
| **SCAR Transition Detector** | 102 | 81 | 83 | 9 | 55.7% | 91.9% | 49.4% | 69.4% |

### Combined Corpora

Pooled across all 1,587 sessions (514 unresolved failures and 1,073 resolved passes).

| Detector | Precision | Recall | FPR | F1 |
| :--- | :---: | :---: | :---: | :---: |
| Tool-Error Only | 41.2% | 29.8% | 20.3% | 34.6% |
| Naive Regex Pushback | 29.8% | 61.7% | 69.7% | 40.2% |
| Static Sentiment | 34.4% | 33.5% | 30.6% | 33.9% |
| Contextual User Pushback Only | 55.9% | 12.1% | 4.6% | 19.8% |
| **SCAR Transition Detector** | 61.3% | 62.6% | 18.9% | 62.0% |

## Token Budget Retention

Evaluated across 12 longitudinal 8-session user timelines (averaging 8,338 subword transcript tokens per user and 2,016 live probe evaluations across 10%, 25%, and 50% token budgets).

### 25% Budget Breakdown

| Memory Architecture | Avg Used / Cap (tok) | Static Fact F1 | Updated Pref F1 | Procedural F1 | Non-Pollution | Precision | Overall F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Store-All FIFO (Recency) | 2050 / 2084 | 0.0% | 7.1% | 15.4% | 100.0% | 50.0% | 8.7% |
| Rolling LLM Summary | 506 / 2084 | 88.4% | 95.7% | 100.0% | 100.0% | 100.0% | 95.7% |
| Atomic Facts (Mem0-style) | 664 / 2084 | 97.9% | 79.2% | 66.7% | 58.3% | 85.9% | 78.7% |
| Single-Turn Importance | 488 / 2084 | 97.9% | 95.8% | 100.0% | 66.7% | 93.8% | 92.1% |
| Static Sentiment Filter | 432 / 2084 | 0.0% | 0.0% | 100.0% | 58.3% | 85.7% | 50.4% |
| Tool-Outcome Only | 70 / 2084 | 0.0% | 0.0% | 66.7% | 100.0% | 100.0% | 35.3% |
| SCAR (No Quarantine / Replay) | 359 / 2084 | 97.9% | 100.0% | 100.0% | 50.0% | 92.8% | 92.2% |
| **SCAR (Full)** | 180 / 2084 | 97.9% | 100.0% | 90.9% | 100.0% | 100.0% | 92.3% |

### Cross-Budget Scaling

| Memory Architecture | 10% Overall F1 (Proc F1) | 25% Overall F1 (Proc F1) | 50% Overall F1 (Proc F1) |
| :--- | :---: | :---: | :---: |
| Store-All FIFO (Recency) | 6.7% (15.4%) | 8.7% (15.4%) | 33.6% (54.5%) |
| Rolling LLM Summary | 93.8% (100.0%) | 95.7% (100.0%) | 95.1% (100.0%) |
| Atomic Facts (Mem0-style) | 82.3% (66.7%) | 78.7% (66.7%) | 81.6% (62.9%) |
| Single-Turn Importance | 91.6% (100.0%) | 92.1% (100.0%) | 92.8% (100.0%) |
| Static Sentiment Filter | 48.3% (95.7%) | 50.4% (100.0%) | 49.1% (95.7%) |
| Tool-Outcome Only | 35.3% (66.7%) | 35.3% (66.7%) | 35.3% (66.7%) |
| SCAR (No Quarantine / Replay) | 92.2% (100.0%) | 92.2% (100.0%) | 92.2% (100.0%) |
| **SCAR (Full)** | 91.6% (90.9%) | 92.3% (90.9%) | 92.3% (90.9%) |

## Mistake Reduction

Evaluated via live tool-calling replay on held-out tasks under a disjoint two-fold split where no training task appears in evaluation and all oracle reward fields are stripped before consolidation.

| Memory System | Held-Out Trials | Mistakes Repeated | Repeat Error Rate | Airline Repeat Rate | Retail Repeat Rate | Avg Memory Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-Shot (Policy + Tools) | 52 | 17 | 32.7% | 23.8% | 38.7% | 0 |
| Raw Episode RAG | 52 | 22 | 42.3% | 47.6% | 38.7% | 353 |
| Atomic Facts (Mem0-style) | 52 | 18 | 34.6% | 47.6% | 25.8% | 448 |
| Tool-Outcome Only | 52 | 16 | 30.8% | 38.1% | 25.8% | 324 |
| SCAR (No Quarantine / Replay) | 52 | 16 | 30.8% | 42.9% | 22.6% | 425 |
| **SCAR (Full)** | 52 | 14 | 26.9% | 38.1% | 19.4% | 419 |

- **Relative Repeat Error Reduction (SCAR vs. Zero-Shot):** **17.6%** (vs. Atomic Facts Mem0: **22.2%**)
- **Non-Regression Control Cohort (30 clean passing tasks):** Zero-Shot 18/30 (60.0%) vs. SCAR 19/30 (63.3%).

