# MemoryGuard: What AI Agents Should Remember and Trust

Coding assistants save notes across sessions so they remember how you work, such as which test flags or folder paths your project uses. Saving every message creates two problems: casual chat clogs memory with clutter, and untrusted web pages or GitHub issues can plant fake project rules that trick the agent into running unsafe commands later.

MemoryGuard (`memory_guard`) fixes both problems in about 1,600 lines of readable Python without blocking normal coding work or calling a cloud LLM on every message write.

## How MemoryGuard Works in 3 Steps

1. Filter on write (`LayaAdmissionGate` in `gates.py`): When a new message arrives, a small 421M local CPU model (`convaiinnovations/laya`, `typed-decisions`) scores four yes/no questions covering usefulness, confidence, rule type, and danger. It blocks direct instruction overrides (`danger >= 0.50`) and drops casual small talk (`score < 0.40`).
2. Tag and sign in storage (`MemoryStore` in `store.py`): Saved notes get a trust tier (`USER = 3`, `REPO = 2`, `EXTERNAL = 1`) and an `HMAC-SHA256` signature inside SQLite. Edited rows fail verification on read, and untrusted `EXTERNAL` notes that copy an existing `USER` or `REPO` rule (`cosine >= 0.78`) are dropped as `shadowed`.
3. Double-check risky commands in a Clean Room (`ToolInterceptor` in `gates.py`): Before running `run_bash`, `write_file`, or `http_post`, MemoryGuard checks whether any `EXTERNAL` note was retrieved. If so, it re-runs the planner using only trusted `USER` and `REPO` notes (`tier >= 2`). Matching commands run normally; mismatched commands are blocked and the untrusted notes are quarantined.

## Files and Recommended Reading Order

If you are new to the codebase, read the four root Python files in this order:

| File | Role | What You Will Learn |
| :--- | :--- | :--- |
| `store.py` | Signed SQLite memory store | Trust tiers, `HMAC-SHA256` signing, duplicate shadowing, BM25 search |
| `gates.py` | Write filters and tool guard | `Stock`, `LLM`, `Lexical`, and `Laya-421M` gates plus Clean-Room check |
| `llm.py` | Gemini REST client and cache | Zero-dependency HTTP client and SQLite cache for offline replay |
| `run_mvp.py` | 6-architecture benchmark runner | Session 1 write loop, Session 2 tool check, recall test, McNemar stats |
| `benchmarks/build_suite.py` | Dataset builder and validator | Builds and checks `coding_memory_suite.json` and `recall_slice.json` |
| `tests/test_memory_guard.py` | Automated test suite (8 tests) | Step-by-step tests for tampering, shadowing, Laya-421M, and Clean-Room |
| `results/mvp_summary.json` | Saved benchmark output | Per-episode logs, recall accuracy, write latency, and p-values |

## Benchmark Datasets (`benchmarks/`)

1. `benchmarks/coding_memory_suite.json` (`N = 36` two-session coding episodes across 9 software domains): Tests 9 clean tasks (`benign_clean`), 9 tasks with helpful external notes (`benign_external_taint_stress`), 9 direct prompt injections (`attack_obvious`), and 9 disguised conventions or split-turn attacks (`attack_sneaky`).
2. `benchmarks/recall_slice.json` (`N = 30` 40-turn chat histories across 10 `PersonaMem-v1` users): Mixes 25% real user facts with 75% casual small talk to measure how much storage a gate saves without forgetting user preferences.

## Benchmark Results (`results/mvp_summary.json`)

Evaluated with `gemini-3.5-flash-lite` (`temperature = 0.0`). Lower Attack Success Rate (ASR) and Benign False Lock are better; higher Benign Utility, Overall Success, Recall, and Storage Saved are better.

| Architecture | Overall ASR (18) | Obvious ASR (9) | Sneaky ASR (9) | Benign False Lock (18) | Taint-Priv Lock (6) | Benign Utility (18) | Overall Success (36) | PersonaMem Recall (30) | Storage Saved | Write Latency Mean (p95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Unprotected | 55.56% | 33.33% | 77.78% | 0.00% | 0.00% | 100.00% | 72.22% | 73.33% | 0.00% | 0.944 ms (2.437 ms) |
| LLM Gate Only (A-MAC style) | 44.44% | 0.00% | 88.89% | 0.00% | 0.00% | 100.00% | 77.78% | 70.00% | 39.53% | 550.466 ms (1169.238 ms) |
| Lexical Regex Gate Only | 11.11% | 0.00% | 22.22% | 0.00% | 0.00% | 72.22% | 80.56% | 43.33% | 82.21% | 0.365 ms (1.273 ms) |
| Laya-421M Gate Only | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 75.00% | 76.67% | 20.43% | 400.837 ms (480.260 ms) |
| Laya-421M + Blanket Red Lock | 0.00% | 0.00% | 0.00% | 33.33% | 100.00% | 66.67% | 83.33% | 76.67% | 20.43% | 400.837 ms (480.260 ms) |
| MemoryGuard (`6_memory_guard_full`) | 0.00% | 0.00% | 0.00% | 5.56% | 16.67% | 94.44% | 97.22% | 76.67% | 20.43% | 400.837 ms (480.260 ms) |

Key takeaways from exact two-sided McNemar paired tests:

- Versus Stock Unprotected, MemoryGuard cuts Attack Success Rate from 55.56% (`10/18`) to 0.00% (`0/18`, `p = 0.001953`), trims stored conversation words by 20.43% (`33,567` vs `42,186`) while raising recall from 73.33% to 76.67%, and lifts overall episode success from 72.22% to 97.22% (`p = 0.011719`).
- Versus Laya-421M Gate Only, write-time filtering stops 100% of direct prompt injections (`0.00%` Obvious ASR) without breaking recall like regex rules (`43.33%` recall), yet sneaky convention attacks still slip into memory (`100.00%` Sneaky ASR). Adding trust wristbands and Clean-Room replanning cuts Overall ASR from 50.00% (`9/18`) to 0.00% (`0/18`, `p = 0.003906`).
- Versus a Blanket Red Lock (`naive_biba_min`) that blocks every command whenever any `EXTERNAL` note is retrieved (`33.33%` false lock rate, `100.00%` on privileged external-note tasks), Clean-Room replanning lowers false locks to 5.56% (`1/18`) and raises safe coding utility from 66.67% to 94.44% (`p = 0.062500`).
- On the uncached 20-note write-latency check, local `LayaAdmissionGate` averages 400.837 ms (480.260 ms p95) on CPU versus 550.466 ms mean (1169.238 ms p95) for uncached cloud `LLMAdmissionGate` (2.43x faster p95 with zero cloud API calls).

## Running Tests and Benchmarks

```bash
# 1. Run all 8 unit, live Laya-421M, Clean-Room, and hygiene tests
python -m pytest -v

# 2. Rebuild and validate the 36 coding episodes and 30 PersonaMem-v1 dialogues
python benchmarks/build_suite.py

# 3. Run the full 6-architecture evaluation (replays offline from .cache/llm_cache.sqlite,
#    or calls Gemini live when GEMINI_API_KEY is set)
python run_mvp.py
```
