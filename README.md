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

## Emotion-Shift Memory

`src/esm/` provides a standalone two-module implementation (**Emotion-Shift Memory**) designed around a plain Markdown rulebook (`MEMORY.md`, capped at 400 subword tokens) and four human-feedback shift detectors (`VADER`, `GoEmotions`, local open-weights `Laya` ModernBERT, and `Gemini 3.5 Flash-Lite`):

- **Silent Failure Rate (`tau2-bench`, `n = 2,624`):** Across all four public `tau2-bench` model logs (Claude 3.7 Sonnet, GPT-4.1, GPT-4.1-mini, and o4-mini), **669 of 869 failed conversations (77.0%)** complete with zero software tool errors (88.3% on Airline and 68.6% on Retail).
- **Stage 1 (Human Shift Detection):** Evaluated on 600 balanced human-annotated `microsoft/WildFeedback` dialogues (`2,709` reaction turns) and 176 multi-turn English `OpenHands` coding sessions:

| Shift Detector | WildFeedback Fixed F1 | WildFeedback 4-Way Acc | OpenHands Unresolved F1 | Latency / Turn |
| :--- | :---: | :---: | :---: | :---: |
| `VADER` (Word Lexicon) | 45.0% | 36.3% | 13.3% | 1.7 ms |
| `GoEmotions` (`roberta-base-go_emotions`) | 49.7% | 41.2% | 24.2% | 22 ms |
| `Laya` (`convaiinnovations/laya` CPU) | **59.3%** | 51.3% | 54.7% | 242 ms |
| `Gemini 3.5 Flash-Lite` | 51.0% | **55.0%** | **72.0%** | 144 ms |

- **Stage 2 (5-Condition `MEMORY.md` Ablation on Unseen Tasks):** Evaluated across a 2-fold disjoint task split of `tau2-bench` Airline and Retail (`59` held-out failure decision points plus `31` routine passing controls under a 400-token `MEMORY.md` cap):

| Memory Condition (`<= 400` tokens) | Held-Out Mistakes | Repeat Error Rate | Airline Rate | Retail Rate | Control Pass |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `No Memory` (Zero-Shot) | 26 / 59 | 44.1% | 50.0% | 40.0% | 87.1% |
| `Routine Chats` (`Calm -> Satisfied`) | 29 / 59 | 49.2% | 62.5% | 40.0% | n/a |
| `Negative-Only` (`VADER` Unresolved) | 23 / 59 | 39.0% | 50.0% | 31.4% | 83.9% |
| `All Chats` (Unfiltered Notes) | 21 / 59 | 35.6% | 50.0% | 25.7% | n/a |
| **Emotion-Shift (`Frustrated -> Fixed`)** | **16 / 59** | **27.1%** | **33.3%** | **22.9%** | **83.9%** |

Emotion-Shift Memory reduces repeat mistakes by **38.5%** relative to `No Memory` (`44.1%` to `27.1%`) and **30.4%** relative to `Negative-Only` (`39.0%` to `27.1%`), improving both Airline (`50.0%` to `33.3%`) and Retail (`40.0%` to `22.9%`).

## Reproduction

```bash
git clone https://github.com/srtux/scar.git
cd scar
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
pytest tests/ -q

# Run core SCAR experiments
python3 experiments/run_all.py

# Download WildFeedback (for Stage 1) and run Emotion-Shift Memory + 300-DPI poster charts
hf download microsoft/WildFeedback sat_dsat_annotation.json --repo-type dataset --local-dir data/wildfeedback
python3 experiments/run_esm.py --stage all --workers 16
python3 experiments/plot_poster_charts.py
```

Running `experiments/run_all.py` executes all three SCAR experiments and generates `results/summary_results.json` and `results/SUMMARY_RESULTS.md`. Running `experiments/run_esm.py` and `experiments/plot_poster_charts.py` writes `results/esm_results.json` and renders 300-DPI `PNG` and `PDF` figures under `results/charts/`.

## Repository Layout

```
scar/
├── data/
│   ├── openhands/                 # Public OpenHands human-feedback coding trajectories (n=275)
│   └── tau2/                      # Public tau2-bench Airline & Retail trajectories + tool schemas (n=2,624)
├── experiments/
│   ├── exp1_failure_detection.py  # Silent failure & transition detector evaluation (n=1,587)
│   ├── exp2_retention_budget.py   # Longitudinal 8-session retention at 10%, 25%, 50% token budgets
│   ├── exp3_repeat_mistakes.py    # 2-fold held-out cross-task tool-calling replay (n=52 + 30 controls)
│   ├── plot_poster_charts.py      # 300-DPI poster chart generator (Charts 1-4 PNG + PDF)
│   ├── run_all.py                 # End-to-end SCAR runner & summary generator
│   └── run_esm.py                 # Stage 1 & Stage 2 Emotion-Shift Memory benchmark runner
├── results/
│   ├── charts/                    # Print-ready 300-DPI PNG and PDF figures (Charts 1-4)
│   ├── esm_results.json           # Stage 1 & Stage 2 Emotion-Shift Memory metrics & sample MEMORY.md
│   ├── exp1_details.json
│   ├── exp2_details.json
│   ├── exp3_details.json
│   ├── summary_results.json
│   └── SUMMARY_RESULTS.md
├── src/
│   ├── esm/
│   │   ├── detectors.py           # VADER, GoEmotions, Laya (ModernBERT), and Flash-Lite shift gates
│   │   └── memory_md.py           # Plain Markdown MEMORY.md store, contrastive extractor & replay verifier
│   └── scar/
│       ├── baselines.py           # FIFO, Rolling Summary, Mem0, Importance, Sentiment, Tool-Only, Raw RAG
│       ├── consolidator.py        # Oracle stripper, struggle-card synthesis & counterfactual replay gate
│       ├── detector.py            # Tool-error, regex, sentiment, pushback & SCAR transition detectors
│       ├── llm.py                 # Rate-limited Gemini REST + SQLite cache client
│       ├── memory.py              # Slot-indexed directive store + contrastive playbook retrieval
│       └── tokens.py              # Deterministic subword token accounting
└── tests/
    ├── test_esm.py                # Unit tests for shift gates, oracle stripping, and 400-token MEMORY.md cap
    ├── test_oss_cleanliness.py    # Verifies zero internal refs, hardcoded keys, or oracle leaks
    └── test_scar_core.py          # Core unit tests for supersession, quarantine, and token caps
```

## License

Apache License 2.0. See `LICENSE` for details.
