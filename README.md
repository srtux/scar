# MemoryGuard: What AI Agents Should Remember and Trust

Coding assistants save notes across sessions to remember project test flags, directory layouts, and personal preferences. Saving raw conversation logs causes two failures: casual small talk clutters retrieved context, and untrusted web pages or GitHub issues can plant fake conventions that steer the agent into unsafe commands later.

MemoryGuard (`memory_guard`) solves both problems in under 2,000 lines of core Python without blocking normal coding work or calling a cloud LLM on message writes.

## Quickstart

You need Python 3.10 or newer and about 1.7 GB of free disk space on the first run for the local `Laya-421M` weights from Hugging Face.

```bash
git clone https://github.com/srtux/memoryguard.git
cd memoryguard

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

On Windows PowerShell, activate the virtual environment with `.venv\Scripts\Activate.ps1`. Open `.env` and set `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey), or uncomment `GOOGLE_CLOUD_PROJECT` to authenticate through Google Cloud Vertex AI. You can also set `GEMINI_MODEL` in `.env` to override the default `gemini-3.5-flash-lite` model.

Run the offline test suite first (no credentials or cloud network access required), then validate the datasets and run the evaluation:

```bash
python -m pytest -v
python benchmarks/build_suite.py
python run_mvp.py
```

The runner reads `.env` automatically and caches model replies in `.cache/llm_cache.sqlite` so offline replays finish in milliseconds.

## How MemoryGuard Works in 3 Steps

1. Filter on write (`LayaAdmissionGate` in `gates.py`): A local 421M CPU model (`convaiinnovations/laya`, `typed-decisions`) scores four yes/no questions covering utility, confidence, rule category, and danger. It vetoes direct instruction overrides first (`danger >= 0.50`) and drops everyday small talk (`score < 0.40`) before storage.
2. Tag and sign in storage (`MemoryStore` in `store.py`): Saved notes get a provenance tier (`USER = 3`, `REPO = 2`, `EXTERNAL = 1`) and a full-row `HMAC-SHA256` signature (including `active` status) using a random 256-bit key. Tampered rows fail verification on read, untrusted `EXTERNAL` copies of `USER` or `REPO` rules are dropped as `shadowed`, and identifier/polarity checks prevent distinct rules from overwriting one another.
3. Verify and recover in a Clean Room (`ToolInterceptor` in `gates.py`): Before running shell commands, file writes, HTTP posts, sensitive file reads (`.env`, `.ssh`, `.pypirc`), or URL/beacon answers, MemoryGuard checks whether any `EXTERNAL` note was retrieved via SQLite `FTS5` BM25. When untrusted notes are present, it replans with `USER` and `REPO` notes alone, compares plans structurally (`shlex`, `ast`, `json`, SQL), quarantines untrusted notes on mismatch, and safely executes the clean trusted plan instead of aborting the user's task.

## Modules

Read the files in this order when exploring the codebase:

| File | Role | Contents |
| :--- | :--- | :--- |
| `store.py` | Signed SQLite + FTS5 store | Trust tiers, full-row `HMAC-SHA256` signing, shadowing, `FTS5` BM25 search |
| `gates.py` | Write gates and Clean-Room | `Stock`, `LLM`, `Lexical`, and `Laya-421M` gates plus structural Clean-Room recovery |
| `llm.py` | Gemini SDK client and cache | `.env` loader, `google-genai` client, `tenacity` retries, SQLite prompt cache |
| `run_mvp.py` | 6-arm benchmark runner | Two-session coding loop, recall test, write-latency check, McNemar p-values |
| `.env.example` | Environment template | Tracked config template for `GEMINI_API_KEY`, `GEMINI_MODEL`, and Vertex AI |
| `benchmarks/build_suite.py` | Dataset builder | Generates and validates the 48 coding episodes and 30 `PersonaMem-v1` histories |
| `tests/test_memory_guard.py` | Automated test suite | Offline unit, live `Laya-421M`, Clean-Room recovery, cache replay, hygiene checks |
| `results/mvp_summary.json` | Saved benchmark output | Verified metrics, per-episode traces, write latencies, and paired p-values |

## Benchmarks

1. Multi-session coding suite (`benchmarks/coding_memory_suite.json`, `N = 48` episodes across 4 balanced cohorts of 12): Tests 12 clean tasks (`benign_clean`), 12 tasks mixing benign external notes with user rules (`benign_external_taint_stress`, including 9 privileged tasks), 12 direct prompt injections (`attack_obvious`, including `.env`/`id_ed25519` reads and markdown beacons), and 12 sneaky convention, split-turn, and credential-read hijacks (`attack_sneaky`). Every episode requires both blocking any hijack and completing the user's real coding task.
2. Conversational recall slice (`benchmarks/recall_slice.json`, `N = 30` 40-turn chat histories across 10 `PersonaMem-v1` personas): Mixes 10 real user fact turns with 30 organic assistant and session-interruption turns per conversation to measure memory bloat savings against 4-choice personal preference recall.

## Results (`results/mvp_summary.json`)

Evaluated with `gemini-3.5-flash-lite` (`temperature = 0.0`) across 48 coding episodes (24 benign, 24 attacked) and 30 `PersonaMem-v1` dialogues. Lower Attack Success Rate (ASR) and Benign False Lock are better; higher utility, success, recall, and storage savings are better.

| Architecture | Overall ASR (24) | Obvious ASR (12) | Sneaky ASR (12) | Benign False Lock (24) | Taint-Priv Lock (9) | Benign Utility (24) | Attacked Utility (24) | Overall Success (48) | PersonaMem Recall (30) | Storage Saved | Write Latency Mean (p95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Unprotected | 50.00% | 16.67% | 83.33% | 0.00% | 0.00% | 100.00% | 50.00% | 75.00% | 73.33% | 0.00% | 0.8 ms (1.4 ms) |
| LLM Gate Only (A-MAC style) | 41.67% | 0.00% | 83.33% | 0.00% | 0.00% | 100.00% | 58.33% | 79.17% | 60.00% | 51.49% | 555 ms (1,550 ms) |
| Lexical Regex Gate Only | 29.17% | 0.00% | 58.33% | 0.00% | 0.00% | 79.17% | 62.50% | 70.83% | 33.33% | 85.85% | 0.2 ms (0.3 ms) |
| Laya-421M Gate Only | 20.83% | 0.00% | 41.67% | 0.00% | 0.00% | 100.00% | 79.17% | 89.58% | 76.67% | 23.02% | 726 ms (893 ms) |
| Laya-421M + Blanket Red Lock | 0.00% | 0.00% | 0.00% | 33.33% | 88.89% | 66.67% | 70.83% | 68.75% | 76.67% | 23.02% | 726 ms (893 ms) |
| MemoryGuard (`6_memory_guard_full`) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% | 100.00% | 76.67% | 23.02% | 726 ms (893 ms) |

Paired exact two-sided McNemar comparisons and latency findings:

- Versus Stock Unprotected, MemoryGuard cuts overall Attack Success Rate from 50.00% (`12/24`) to 0.00% (`0/24`, `p = 0.000488`), raises task completion under active attack from 50.00% to 100.00% (`p = 0.000488`), lifts overall episode success from 75.00% (`36/48`) to 100.00% (`48/48`, `p = 0.000488`), and trims stored conversation words by 23.02% (`37,080` vs `48,171`) while lifting `PersonaMem-v1` recall from 73.33% (`22/30`) to 76.67% (`23/30`).
- Versus Cloud LLM Gate Only (`41.67%` ASR, `p = 0.001953`) and Lexical Regex Gate Only (`29.17%` ASR, `33.33%` recall, `p = 0.000122` on overall success), `Laya-421M` blocks 100% of direct prompt injections (`0.00%` Obvious ASR) without hurting personal recall, while Clean-Room counterfactual replanning catches the sneaky convention and split-turn hijacks (`83.33%` Sneaky ASR on Stock/LLM, `41.67%` on Laya-Only) that slip past write-only filters.
- Versus Blanket Red Lock (`5_laya_naive_biba`), which falsely locks 33.33% of benign episodes (`88.89%` of privileged tasks with benign external notes), structural Clean-Room verification and recovery eliminate false locks (`0.00%`), raising benign utility from 66.67% to 100.00% (`p = 0.007812`), attacked task utility from 70.83% to 100.00% (`p = 0.015625`), and overall episode success from 68.75% (`33/48`) to 100.00% (`48/48`, `p = 0.000061`).
- On the uncached 20-note write-latency benchmark, local CPU inference with `Laya-421M` achieves an 893 ms p95 tail versus 1,550 ms p95 for uncached cloud LLM gating (1.74x lower p95 tail latency with zero cloud API calls).
