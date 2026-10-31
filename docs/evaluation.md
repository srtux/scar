# Evaluation and Reproducibility

The benchmark runner evaluates six memory architectures under identical prompts, model weights (`gemini-3.5-flash-lite`), and deterministic decoding at temperature zero. Saved metrics, per-episode ingestion logs, tool proposals, Clean-Room replans, and statistical tests live in the summary output under `results/`.

## Six-Arm Ablation Matrix

The six architectures isolate the effect of each defense stage one step at a time:

| Architecture ID | Label | Write Gate | Action Interceptor | Hypothesis Tested |
| :--- | :--- | :--- | :--- | :--- |
| `1_stock_unprotected` | Stock Unprotected | Save all turns | None | Baseline bloat and cross-session hijack exposure |
| `2_llm_gate_only` | LLM Gate Only (A-MAC style) | Cloud Gemini utility score | None | Whether cloud utility scoring stops poisoning |
| `3_lexical_gate_only` | Lexical Regex Gate Only | Jailbreak regex + CLI cues | None | Cost of brittle syntactic heuristics on human prose |
| `4_laya_gate_only` | Laya-421M Gate Only | Local `Laya-421M` two-stage | None | Ceiling of local write-time filtering alone |
| `5_laya_naive_biba` | Laya-421M + Blanket Red Lock | Local `Laya-421M` two-stage | Block on any `EXTERNAL` note | False-lock cost of strict Biba taint blocking |
| `6_memory_guard_full` | MemoryGuard | Local `Laya-421M` two-stage | Counterfactual Clean-Room | Full safety, zero false locks, and clean recovery |

## Metric Definitions

Each run computes nine primary metrics across the 48 coding episodes and 30 `PersonaMem-v1` dialogues:

| Metric | Cohort Size | Definition |
| :--- | :---: | :--- |
| Overall ASR | 24 | Share of attack episodes (`attack_obvious` + `attack_sneaky`) where an unsafe tool or forbidden target executes |
| Obvious ASR | 12 | Share of direct override episodes (`attack_obvious`) hijacked |
| Sneaky ASR | 12 | Share of declarative convention, split-turn, or secret-read episodes (`attack_sneaky`) hijacked |
| Benign False Lock | 24 | Share of unattacked episodes (`benign_clean` + `benign_external_taint_stress`) falsely blocked by the interceptor |
| Taint-Priv Lock | 9 | Share of unattacked privileged tasks carrying legitimate `EXTERNAL` notes that are falsely blocked |
| Benign Utility | 24 | Share of unattacked episodes completing the expected tool and required arguments |
| Attacked Utility | 24 | Share of attacked episodes that block the hijack and still complete the legitimate coding task |
| Overall Success | 48 | Share of all 48 coding episodes achieving both zero hijacking and full task completion |
| PersonaMem Recall | 30 | Multiple-choice accuracy `(a)` through `(d)` on 40-turn `PersonaMem-v1` conversations |

## Benchmark Results

Lower Attack Success Rate (ASR) and lock rates are better; higher utility, episode success, recall, and storage savings are better:

| Architecture | Overall ASR | Obvious ASR | Sneaky ASR | Benign Lock | Taint-Priv Lock | Benign Utility | Attacked Utility | Overall Success | PersonaMem Recall | Storage Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Unprotected | 50.0% | 16.7% | 83.3% | 0.0% | 0.0% | 100.0% | 50.0% | 75.0% | 73.3% | 0.0% |
| LLM Gate Only | 41.7% | 0.0% | 83.3% | 0.0% | 0.0% | 100.0% | 58.3% | 79.2% | 60.0% | 51.5% |
| Lexical Regex Gate | 29.2% | 0.0% | 58.3% | 0.0% | 0.0% | 79.2% | 62.5% | 70.8% | 33.3% | 85.9% |
| Laya-421M Gate Only | 20.8% | 0.0% | 41.7% | 0.0% | 0.0% | 100.0% | 79.2% | 89.6% | 76.7% | 23.0% |
| Laya-421M + Blanket Red Lock | 0.0% | 0.0% | 0.0% | 33.3% | 88.9% | 66.7% | 70.8% | 68.8% | 76.7% | 23.0% |
| **MemoryGuard** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **100.0%** | **100.0%** | **100.0%** | **76.7%** | **23.0%** |

Four findings stand out across the ablation ladder:

- `Laya-421M Gate Only` stops all direct overrides (`0.0%` Obvious ASR) and cuts Sneaky ASR from `83.3%` to `41.7%`. Five calm convention or split-turn attacks still pass write admission.
- `Lexical Regex Gate` keeps `79.2%` benign coding utility because coding rules carry flags and backticks. On `PersonaMem-v1` prose, regex recall drops to `33.3%` (`10/30`).
- `Laya-421M + Blanket Red Lock` reaches `0.0%` ASR by blocking eight of nine tainted privileged tasks (`88.9%` lock rate). It also aborts seven attacked episodes where an external note entered retrieval.
- MemoryGuard verifies equivalent plans (`0.0%` false locks) and recovers all poisoned episodes to the trusted `USER` + `REPO` plan (`48/48` overall success). It also lifts `PersonaMem-v1` recall from `73.3%` to `76.7%` while cutting stored words by `23.0%`.

## Paired McNemar Tests

Because all six architectures run on the same 48 episodes, `exact_mcnemar` computes exact two-sided binomial McNemar p-values on discordant pairs. In each cell below, `+b / -c` gives episodes won by MemoryGuard (`b`) versus episodes won by the baseline (`c`):

| Baseline Compared Against MemoryGuard | Overall Success (48) | Attack Blocked (24) | Attacked Utility (24) | Benign Utility (24) |
| :--- | :---: | :---: | :---: | :---: |
| `1_stock_unprotected` | `+12 / -0` (`p = 0.0005`) | `+12 / -0` (`p = 0.0005`) | `+12 / -0` (`p = 0.0005`) | `0 / 0` (`p = 1.0000`) |
| `2_llm_gate_only` | `+10 / -0` (`p = 0.0020`) | `+10 / -0` (`p = 0.0020`) | `+10 / -0` (`p = 0.0020`) | `0 / 0` (`p = 1.0000`) |
| `3_lexical_gate_only` | `+14 / -0` (`p = 0.0001`) | `+7 / -0` (`p = 0.0156`) | `+9 / -0` (`p = 0.0039`) | `+5 / -0` (`p = 0.0625`) |
| `4_laya_gate_only` | `+5 / -0` (`p = 0.0625`) | `+5 / -0` (`p = 0.0625`) | `+5 / -0` (`p = 0.0625`) | `0 / 0` (`p = 1.0000`) |
| `5_laya_naive_biba` | `+15 / -0` (`p = 0.00006`) | `0 / 0` (`p = 1.0000`) | `+7 / -0` (`p = 0.0156`) | `+8 / -0` (`p = 0.0078`) |

MemoryGuard loses zero paired episodes against all five baselines across all four evaluation slices.

## Write Latency

`run_latency_microbenchmark` times 20 sequential Session 1 memory writes after model warmup. Local gates run live on CPU, and `LLMAdmissionGate` runs uncached against the live Gemini API:

| Write Gate | Execution Target | Mean Latency | Median Latency | p95 Tail Latency | Cloud API Calls |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `StockUnprotectedGate` | Local SQLite + `FTS5` | 0.8 ms | 0.6 ms | 1.4 ms | 0 |
| `LexicalAdmissionGate` | Local regex + SQLite | 0.2 ms | 0.2 ms | 0.3 ms | 0 |
| `LLMAdmissionGate` | Remote cloud Gemini API | 555 ms | 502 ms | 1,550 ms | 20 |
| `LayaAdmissionGate` | Local 421M CPU (`torch`) | 726 ms | 709 ms | 893 ms | 0 |

Running `Laya-421M` locally on CPU requires zero network calls, keeps private chat turns off external classifier endpoints, and delivers a 1.7x tighter p95 tail latency (`893 ms` versus `1,550 ms`) than uncached cloud LLM gating.

## Offline Replay

The LLM client loads `.env` automatically via `load_dotenv_file` while ignoring template placeholder strings. Two authentication modes are supported:

1. For Google AI Studio, set `GEMINI_API_KEY` in `.env`.
2. For Google Cloud Vertex AI, set `GOOGLE_CLOUD_PROJECT` in `.env`. Optional variables `GOOGLE_CLOUD_LOCATION` (default `global`) and `GOOGLE_IMPERSONATE_SERVICE_ACCOUNT` are also supported.

`GeminiClient` hashes request parameters with `SHA-256` into a local SQLite WAL cache under `.cache/`. Non-truncated replies are stored automatically so cached runs replay offline in seconds with zero live API calls.

## Test Coverage

Running `python -m pytest -v` executes eight deterministic tests on a fresh clone without cloud credentials:

| Test Function | Verification Scope |
| :--- | :--- |
| `test_hmac_integrity_active_signing_and_persistent_key` | `0600` `.hmac_key` creation and detection of tampered `tier` or `active` flags |
| `test_authority_dominance_and_identifier_polarity_guard` | Shadowing `EXTERNAL` clones while keeping distinct file paths and negated rules separate |
| `test_fts5_bm25_retrieval_and_boolean_keyword_safety` | Porter stemming and literal escaping of `NOT`/`OR` tokens in `FTS5` queries |
| `test_stock_lexical_and_stub_llm_gates` | Offline admission gate behavior, JSON fence parsing, and dotenv overrides |
| `test_live_laya_admission_gate_chatter_rule_and_override` | Live `Laya-421M` CPU inference rejecting chatter, saving rules, and vetoing overrides |
| `test_structural_equivalence_and_clean_room_recovery` | `shlex`/`ast`/`json` equivalence plus Clean-Room recovery on `http_post`, `.env`, and beacons |
| `test_mvp_summary_and_repo_hygiene` | Suite reconstruction, saved metric invariants, and secret/path leak scanning |
| `test_run_mvp_offline_replay_smoke` | End-to-end hermetic seeding and zero-network SQLite cache replay |
