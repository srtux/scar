# SCAR: Struggle-Conditioned Agent Retention for Long-Term Human-AI Collaboration

SCAR is an adaptive memory consolidation architecture for personal AI agents. It uses transitions from multi-turn failure to verified resolution (collaborative struggles) as a trigger for long-term memory consolidation.

---

## Abstract

When humans and personal AI assistants collaborate across months of interaction, conversation history grows too large to search efficiently. Saving every turn clutters memory with small talk and outdated preferences, while aggressive summarization destroys specific technical details. Most memory systems select entries by recency or single-turn importance ratings. At the same time, naive emotion-based filtering over-retains angry complaints and misses calm instructions.

We present SCAR (Struggle-Conditioned Agent Retention), an adaptive memory architecture that uses transitions from failure to verified resolution as a trigger for memory consolidation. SCAR monitors user pushback (rephrased queries, explicit corrections) and agent tool errors to identify multi-turn collaborative struggles that end in confirmed success. Calm facts and standing instructions route directly to memory, automatically updating when preferences change. Unresolved failures are excluded from long-term memory so past errors never pollute future prompts. Solved problems become practical rules, tested on the original failure before saving.

We evaluated SCAR on 400 multi-turn interactive trajectories from $\tau^2$-bench across Claude 3.7 Sonnet and GPT-4.1. We found that 91.0% of failures are silent policy errors where tools report success, rendering tool-outcome checkers ineffective (9.0% recall), while SCAR's pushback detector catches up to 78.0% of failures. Under a 25% token budget, SCAR achieved 100% procedural retention compared to 35.0% for atomic fact baselines, eliminating repeat mistakes across 48 follow-up evaluation trials. These results demonstrate that failure-to-resolution transitions provide an effective learning signal for long-term human-AI collaboration.

---

## Core Architecture

SCAR splits memory into two complementary pathways:

```
[User & Agent Interaction Stream]
                │
                ├─────────────────────────────────────────┐
                ▼                                         ▼
   [Calm Facts & Directives]               [Collaborative Struggles]
   (e.g., "I use Python 3.11")             (Tool errors + User pushback)
                │                                         │
                ▼                                         ▼
   [Keyed Directives Store]                 [Resolution Verification]
   • Key-based supersession                 • Did the struggle reach success?
   • Overwrites older preferences           • Discards unverified / abandoned errors
                │                                         │
                │                                         ▼
                │                           [Contrastive Playbook Card]
                │                           • Trigger: Context condition
                │                           • Avoid: Action that failed
                │                           • Instead: Action that resolved it
                │                                         │
                └───────────────────┬─────────────────────┘
                                    ▼
                      [Prompt Injection & Retrieval]
```

1. **Keyed Directives (Utility Path):** Stores declarative facts and user preferences. When a preference changes, key-based supersession overwrites the obsolete entry, preventing conflicting memories from competing during retrieval.
2. **Contrastive Playbook Cards (Struggle Path):** When an agent struggles through a multi-turn failure and successfully resolves it, an offline pass synthesizes a "do this, not that" rule.
3. **Quarantine for Unresolved Failures:** If a struggle was abandoned or unresolved, it is excluded from memory so the agent never imitates its own failed attempts.

---

## Measured Experimental Results

All experiments are evaluated on public $\tau^2$-bench trajectories across 400 complete multi-turn sessions (200 Claude 3.7 Sonnet, 200 GPT-4.1).

### 1. Silent Failures & Error Detection

| Detection Method | Claude 3.7 Sonnet Recall | GPT-4.1 Recall | Combined Recall (n=400) |
| :--- | :---: | :---: | :---: |
| **Tool Error Codes Alone** | 9.0% (9/100) | 9.1% (8/88) | **9.0% (17/188)** |
| **User Pushback Alone** | **78.0% (78/100)** | **44.3% (39/88)** | **62.2% (117/188)** |
| **SCAR Dual Detector** | **78.0% (78/100)** | **51.1% (45/88)** | **65.4% (123/188)** |

*   **91.0% of failures (171/188)** had ZERO tool errors (silent policy errors).
*   User pushback recovered **62.0%** of silent failures that tool monitoring missed.

### 2. Memory Retention Under a 25% Token Budget

| Memory Policy | Fact Recall | Procedural Rule Recall | Overall Retention |
| :--- | :---: | :---: | :---: |
| **Store-All FIFO (Truncation)** | 50.0% | 100.0% | 75.0% |
| **Atomic Facts (Mem0)** | 100.0% | 35.0% | 67.5% |
| **Tool-Outcome Only** | 20.0% | 14.3% | 17.1% |
| **SCAR (Dual Key + Playbook)** | **100.0%** | **100.0%** | **100.0%** |

### 3. Repeat Mistake Reduction in Follow-up Tasks

| System | Follow-up Trials | Mistakes Repeated | Repeat Error Rate |
| :--- | :---: | :---: | :---: |
| **Zero-Shot (No Memory)** | 48 | 20 | **41.7%** |
| **Raw History (FIFO)** | 48 | 20 | **41.7%** |
| **Atomic Facts (Mem0)** | 48 | 20 | **41.7%** |
| **SCAR (Contrastive Rule)** | 48 | 0 | **0.0%** |

*   **Absolute Error Rate Reduction (SCAR vs. Mem0):** **41.7%**

---

## Quickstart

### Installation

```bash
git clone https://github.com/srtux/scar.git
cd scar
pip install -r requirements.txt
```

### Reproducing All Experiments

Run the end-to-end evaluation suite:

```bash
python3 experiments/run_all.py
```

The script will re-evaluate all 400 trajectories, print the summary tables, and save results to `results/summary_results.json` and `results/SUMMARY_RESULTS.md`.

---

## Repository Structure

```
scar/
├── data/
│   └── tau2/              # Public tau2-bench benchmark trajectory logs
├── experiments/
│   ├── exp1_failure_detection.py
│   ├── exp2_retention_budget.py
│   ├── exp3_repeat_mistakes.py
│   └── run_all.py         # End-to-end execution runner
├── results/
│   ├── summary_results.json
│   └── SUMMARY_RESULTS.md
├── src/
│   └── scar/
│       ├── __init__.py
│       ├── detector.py    # Pushback and tool failure detection
│       └── memory.py      # Dual-store memory (Keyed Directives + Contrastive Cards)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## License

Apache License 2.0. See `LICENSE` for details.
