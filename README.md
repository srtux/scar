# SCAR

Struggle-Conditioned Agent Retention (SCAR) is an adaptive memory consolidation architecture for long-term human-AI collaboration that triggers procedural rule synthesis on verified transitions from struggle to resolution while quarantining unresolved failures.

## Abstract

When humans and personal AI assistants collaborate across months of interaction, conversation history grows too large to search efficiently. Saving full transcripts clutters memory with small talk and outdated preferences, while aggressive summarization drops specific technical constraints. Existing memory systems select entries by recency or single-turn importance, whereas static sentiment filters retain angry complaints from failed sessions and miss calm instructions.

We present SCAR (Struggle-Conditioned Agent Retention), an adaptive memory architecture that uses transitions from failure to verified resolution as a consolidation signal. SCAR combines tool execution outcomes with contextual user pushback to separate resolved struggles from unresolved failures. Calm facts and standing instructions route directly into a slot-indexed directive store that overwrites superseded preferences. Unresolved struggles are quarantined so failed trajectories do not pollute future prompts, while resolved struggles become contrastive playbook rules verified via counterfactual replay on the original failure prefix before storage.

Across 1,587 multi-turn trajectories from tau2-bench (n = 1,312) and human-rated OpenHands coding sessions (n = 275), 81.4% of tau2-bench failures occur with zero tool errors, and SCAR reaches 62.0% failure detection F1 versus 34.6% for tool-error monitoring and 33.9% for static sentiment. At a 25% token budget, SCAR reaches 92.3% overall retention F1, 100% preference-update F1, and 90.9% procedural F1 (+24.2 points over atomic fact extraction) with zero quarantine pollution. On held-out cross-task replay, SCAR lowers repeat mistake rates to 26.9% versus 32.7% zero-shot, 34.6% atomic facts, and 42.3% raw trajectory RAG, cutting retail repeat errors by 50.0%.

## Architecture

SCAR separates long-term agent memory into two pathways governed by a struggle-to-resolution state machine:

```
[Multi-Turn Human & Agent Trajectory]
                  |
                  +-------------------------------------------+
                  v                                           v
     [Calm Facts & Preferences]                [Tool Errors + User Pushback]
                  |                                           |
                  v                                           v
   [Slot-Indexed Directive Store]             [Transition State Classifier]
   - Canonical slot normalization             - CLEAN_SUCCESS -> no-op
   - Overwrites superseded values             - UNRESOLVED_FAILURE -> quarantine
                  |                           - RESOLVED_STRUGGLE -> synthesize
                  |                                           |
                  |                                           v
                  |                             [Counterfactual Replay Gate]
                  |                             - Re-runs model at failure turn t*
                  |                             - Stores card if mistake is avoided
                  |                                           |
                  +---------------------+---------------------+
                                        v
                   [Token-Budgeted Hybrid Retrieval (<= B tokens)]
```

1. **Slot-Indexed Directives:** Calm user facts and standing preferences route into a canonical slot map (`preferred_seat`, `package_manager`, `refund_destination`) that overwrites outdated entries in place rather than accumulating conflicting historical statements.
2. **Contrastive Playbook Cards:** Multi-turn episodes that exhibit mid-session friction (tool execution errors or contextual user corrections) and finish in verified resolution are distilled into structured `Trigger / Avoid / Do Instead / Verification` rules.
3. **Quarantine and Counterfactual Replay:** Sessions classified as unresolved failures are excluded from consolidation so failed tool invocations are not retrieved as few-shot examples. Candidate playbook cards must prevent the original mistake when replayed on the historical failure prefix before entering long-term memory.

## Methodology

All three experiments execute live model inference (`gemini-3.8-flash`, auxiliary `gemini-3.5-flash-lite`) and dense vector retrieval (`gemini-embedding-001`) with three anti-leakage controls:

1. **Oracle Stripping:** The preprocessing pipeline removes benchmark reward scores, evaluation criteria, and human feedback labels before failure detection and memory consolidation.
2. **True Subword Token Budgets:** Memory retrieval enforces strict subword token caps at 10%, 25%, and 50% of each user's raw multi-session transcript length.
3. **Disjoint Cross-Task Splits:** Experiment 3 partitions `tau2-bench` tasks into two non-overlapping folds across Airline (50 tasks) and Retail (114 tasks), consolidating memories on training tasks and replaying native tool calls at the first state-mutating turn on held-out test tasks.

## Failure Detection

Across 1,312 `tau2-bench` sessions (400 Airline and 912 Retail across Claude 3.7 Sonnet and GPT-4.1), **328 of 403 failed sessions (81.4%)** complete with zero tool execution errors (91.0% on Airline and 73.0% on Retail). Conversely, 125 passing sessions encounter mid-session tool errors and recover before completion (95 classified as resolved by SCAR). On 275 human-rated `OpenHands` coding sessions (111 negative and 164 positive), 29.7% of negative sessions contain zero command errors while 93 of 164 positive sessions recover from mid-session command errors.

| Detector | tau2-bench F1 (n=1,312) | OpenHands F1 (n=275) | Combined Precision | Combined Recall | Combined FPR | Combined F1 (n=1,587) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Tool-Error Only | 24.9% | 55.3% | 41.2% | 29.8% | 20.3% | 34.6% |
| Naive Regex Pushback | 42.3% | 21.2% | 29.8% | 61.7% | 69.7% | 40.2% |
| Static Sentiment | 37.0% | 12.5% | 34.4% | 33.5% | 30.6% | 33.9% |
| Contextual Pushback Only | 14.4% | 33.3% | 55.9% | 12.1% | 4.6% | 19.8% |
| **SCAR Transition Detector** | **59.1%** | **69.4%** | **61.3%** | **62.6%** | **18.9%** | **62.0%** |

## Budgeted Retention

Evaluated across 12 longitudinal 8-session user timelines (averaging 8,338 raw subword tokens per user and 2,016 live probe evaluations across 10%, 25%, and 50% budgets). Each timeline interleaves initial profile facts, two resolved domain struggles, routine lookup sessions, an unresolved policy-violation session, explicit mid-timeline preference updates, and a negative human coding trajectory.

| Memory Architecture (25% Cap = 2,084 tok) | Avg Used (tok) | Static Fact F1 | Updated Pref F1 | Procedural F1 | Non-Pollution | Precision | Overall F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Store-All FIFO (Recency) | 2,050 | 0.0% | 7.1% | 15.4% | 100.0% | 50.0% | 8.7% |
| Rolling LLM Summary | 506 | 88.4% | 95.7% | 100.0% | 100.0% | 100.0% | 95.7% |
| Atomic Facts (Mem0-style) | 664 | 97.9% | 79.2% | 66.7% | 58.3% | 85.9% | 78.7% |
| Single-Turn Importance | 488 | 97.9% | 95.8% | 100.0% | 66.7% | 93.8% | 92.1% |
| Static Sentiment Filter | 432 | 0.0% | 0.0% | 100.0% | 58.3% | 85.7% | 50.4% |
| Tool-Outcome Only | 70 | 0.0% | 0.0% | 66.7% | 100.0% | 100.0% | 35.3% |
| SCAR (No Quarantine / Replay) | 359 | 97.9% | 100.0% | 100.0% | 50.0% | 92.8% | 92.2% |
| **SCAR (Full)** | **180** | **97.9%** | **100.0%** | **90.9%** | **100.0%** | **100.0%** | **92.3%** |

Across budgets (10%, 25%, and 50%), SCAR holds 91.6% to 92.3% overall F1 and 100.0% precision using an average of 180 tokens (2.8x smaller than Rolling Summary and 3.7x smaller than Atomic Facts). Removing quarantine and replay (`SCAR No Quarantine / Replay`) causes 50.0% pollution on unresolved-failure traps (6 of 12 probes endorsing invalid overrides), matching the failure mode seen in `Atomic Facts` (58.3% non-pollution) and `Single-Turn Importance` (66.7% non-pollution).

## Mistake Reduction

Evaluated on 52 held-out first-mutation failure decision points (21 Airline and 31 Retail) plus 30 clean passing control turns under a 450-token memory retrieval cap across disjoint training and evaluation task splits:

| Memory System | Held-Out Trials | Mistakes Repeated | Repeat Error Rate | Airline Rate | Retail Rate | Tool-Error Trap Rate | Avg Memory Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-Shot (Policy + Tools) | 52 | 17 | 32.7% | 23.8% | 38.7% | 31.4% | 0 |
| Raw Episode RAG | 52 | 22 | 42.3% | 47.6% | 38.7% | 37.1% | 353 |
| Atomic Facts (Mem0-style) | 52 | 18 | 34.6% | 47.6% | 25.8% | 28.6% | 449 |
| Tool-Outcome Only | 52 | 16 | 30.8% | 38.1% | 25.8% | 28.6% | 324 |
| SCAR (No Quarantine / Replay) | 52 | 16 | 30.8% | 42.9% | 22.6% | 22.9% | 425 |
| **SCAR (Full)** | **52** | **14** | **26.9%** | **38.1%** | **19.4%** | **17.1%** | **419** |

Retrieving unquarantined episodes (`Raw Episode RAG`, 42.3%) or flat extracted statements (`Atomic Facts`, 34.6%) increases overall mistakes above zero-shot (32.7%) by surfacing failed historical actions in context. SCAR's quarantine and counterfactual replay gate filters 77 candidate cards down to 65 verified cards, lowering repeat errors to 26.9% overall (17.6% relative reduction vs. zero-shot, 22.2% vs. Atomic Facts, and 36.4% vs. Raw Episode RAG) while preserving passing control accuracy (63.3% vs. 60.0% zero-shot).

## Limitations

Two empirical trade-offs warrant attention:

- **Medium-horizon summarization:** On 8-session timelines where a 506-token Rolling Summary fits well inside a 2,084-token budget, Rolling Summary achieves 95.7% overall F1 compared to 92.3% for SCAR, though Rolling Summary loses more early static facts (88.4% vs. 97.9% F1) and uses 2.8x as many tokens.
- **Domain card specificity:** In Experiment 3, SCAR cuts Retail repeat mistakes in half (38.7% to 19.4%) and reduces tool-parameter errors from 31.4% to 17.1%, whereas on the smaller 25-task Airline folds, retrieved cards still over-trigger cautions on borderline eligibility cases (38.1% vs. 23.8% zero-shot, compared to 47.6% for Raw RAG and Atomic Facts). Tighter precondition scoping during card retrieval is an active next step.

## Reproduction

```bash
git clone https://github.com/srtux/scar.git
cd scar
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
pytest tests/ -q
python3 experiments/run_all.py
```

Running `experiments/run_all.py` executes all three experiments, writes per-trial diagnostics to `results/exp1_details.json`, `results/exp2_details.json`, and `results/exp3_details.json`, and generates `results/summary_results.json` and `results/SUMMARY_RESULTS.md`.

## Repository Layout

```
scar/
├── data/
│   ├── openhands/                 # Public OpenHands human-feedback coding trajectories (n=275)
│   └── tau2/                      # Public tau2-bench Airline & Retail trajectories + tool schemas (n=1,312)
├── experiments/
│   ├── exp1_failure_detection.py  # Silent failure & transition detector evaluation (n=1,587)
│   ├── exp2_retention_budget.py   # Longitudinal 8-session retention at 10%, 25%, 50% token budgets
│   ├── exp3_repeat_mistakes.py    # 2-fold held-out cross-task tool-calling replay (n=52 + 30 controls)
│   └── run_all.py                 # End-to-end runner & summary generator
├── results/
│   ├── exp1_details.json
│   ├── exp2_details.json
│   ├── exp3_details.json
│   ├── summary_results.json
│   └── SUMMARY_RESULTS.md
├── src/scar/
│   ├── baselines.py               # FIFO, Rolling Summary, Mem0, Importance, Sentiment, Tool-Only, Raw RAG
│   ├── consolidator.py            # Oracle stripper, struggle-card synthesis & counterfactual replay gate
│   ├── detector.py                # Tool-error, regex, sentiment, pushback & SCAR transition detectors
│   ├── llm.py                     # Rate-limited Gemini REST + SQLite cache client
│   ├── memory.py                  # Slot-indexed directive store + contrastive playbook retrieval
│   └── tokens.py                  # Deterministic subword token accounting
└── tests/
    ├── test_oss_cleanliness.py    # Verifies zero internal refs, hardcoded keys, or oracle leaks
    └── test_scar_core.py          # Core unit tests for supersession, quarantine, and token caps
```

## License

Apache License 2.0. See `LICENSE` for details.
