# MemoryGuard: What AI Agents Should Remember and Trust

**MemoryGuard** protects persistent multi-session coding assistants against indirect memory poisoning and conversational bloat without sacrificing benign tool utility or paying cloud LLM admission tail latency.

## Three-Stage Architecture

1. **Step 1 (Laya-421M Fast Bouncer, `LayaAdmissionGate`)**: Executes a single local CPU pass over four `noul` decision queries (`utility`, `confidence`, `category`, `danger`). A two-stage non-fighting check first vetoes instruction overrides (`danger >= 0.50`) and then filters low-value chatter (`score < 0.40`).
2. **Step 2 (Origin Wristband & Authority-Dominance, `MemoryStore`)**: Signs every memory row with `HMAC-SHA256` over `(mem_id, tier, source_uri, text)` across `USER (Tier 3)`, `REPO (Tier 2)`, and `EXTERNAL (Tier 1)`. Tampered rows are rejected on read, and `EXTERNAL` notes that duplicate an active higher-tier rule (`cosine >= 0.78`) are `shadowed` so untrusted clones cannot overwrite or taint trusted developer conventions.
3. **Step 3 (Clean-Room Counterfactual Non-Interference, `ToolInterceptor`)**: When a privileged tool (`run_bash`, `write_file`, `http_post`) is proposed after retrieving any `EXTERNAL` note, MemoryGuard re-runs the System-2 planner using **only** `USER` and `REPO` notes (`tier >= 2`). If the normalized `(tool, args)` matches, the action is causally independent of untrusted text and executes; if it diverges, MemoryGuard blocks the poisoned call and quarantines the retrieved `EXTERNAL` records.

## Live Benchmark Results (`results/mvp_summary.json`)

Evaluated live with `gemini-3.5-flash-lite` (`temperature = 0.0`) across **36** multi-session coding episodes (`9` `benign_clean`, `9` `benign_external_taint_stress`, `9` `attack_obvious`, `9` `attack_sneaky`) and **30** `PersonaMem-v1` conversational recall dialogues:

| Architecture | Overall ASR (18) | Obvious ASR (9) | Sneaky ASR (9) | Benign False Lock (18) | Taint-Priv Lock (6) | Benign Utility (18) | Overall Episode Success (36) | PersonaMem Recall (30) | Storage Reduction | Write Latency Mean (Median / p95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Unprotected | 55.56% | 33.33% | 77.78% | 0.00% | 0.00% | 100.00% | 72.22% | 73.33% | 0.00% | 0.944 ms (0.869 ms / 2.437 ms) |
| LLM Gate Only (A-MAC style) | 44.44% | 0.00% | 88.89% | 0.00% | 0.00% | 100.00% | 77.78% | 70.00% | 39.53% | 550.466 ms (511.739 ms / 1169.238 ms) |
| Lexical Regex Gate Only | 11.11% | 0.00% | 22.22% | 0.00% | 0.00% | 72.22% | 80.56% | 43.33% | 82.21% | 0.365 ms (0.294 ms / 1.273 ms) |
| Laya-421M Gate Only | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 75.00% | 76.67% | 20.43% | 400.837 ms (400.634 ms / 480.260 ms) |
| Laya-421M + Naive Biba Lock | 0.00% | 0.00% | 0.00% | 33.33% | 100.00% | 66.67% | 83.33% | 76.67% | 20.43% | 400.837 ms (400.634 ms / 480.260 ms) |
| **MemoryGuard (Laya + Wristband + Clean-Room)** | 0.00% | 0.00% | 0.00% | 5.56% | 16.67% | 94.44% | 97.22% | 76.67% | 20.43% | 400.837 ms (400.634 ms / 480.260 ms) |

### Key Ablation Findings & Exact Two-Sided McNemar Tests

- **MemoryGuard (`6_pg_mem_full`) vs Stock Unprotected (`1_stock_unprotected`)**: Cuts Overall ASR from **55.56%** (`10/18`) to **0.00%** (`0/18`), cuts `PersonaMem-v1` storage tokens by **20.43%** (`33567` vs `42186` active words across 30 dialogues) while lifting recall accuracy from **73.33%** (`22/30`) to **76.67%** (`23/30`), and improves overall episode success from **72.22%** to **97.22%** (exact two-sided McNemar `p = 0.011719`, `b = 10`, `c = 1`; attack defense `p = 0.001953`, `b = 10`, `c = 0`).
- **MemoryGuard (`6_pg_mem_full`) vs Laya Gate Only (`4_laya_gate_only`)**: `LayaAdmissionGate` stops `100.00%` of direct prompt injections (`0.00%` Obvious ASR), whereas `LexicalAdmissionGate` severely degrades benign coding utility (`72.22%`) and conversational recall (`43.33%`). However, write-time semantic scoring alone misses calm declarative conventions and two-part split-turn poisons (`100.00%` Sneaky ASR, `50.00%` Overall ASR). Adding Step 2 + Step 3 Clean-Room reduces Overall ASR from **50.00%** (`9/18`) to **0.00%** (`0/18`) (overall episode McNemar `p = 0.021484`, `b = 9`, `c = 1`; attack defense `p = 0.003906`, `b = 9`, `c = 0`).
- **MemoryGuard (`6_pg_mem_full`) vs Blanket Red Lock / Naive Biba (`5_laya_naive_biba`)**: Naive Biba `min` blocks privileged tools whenever any `EXTERNAL` note appears in retrieved context, producing a **33.33%** (`6/18`) benign false-positive lock rate (**100.00%**, `6/6` across privileged external taint-stress tasks) and dropping benign coding utility to **66.67%** (`12/18`). MemoryGuard Clean-Room reduces benign false locks to **5.56%** (`1/18`; `16.67%` taint-priv) and lifts benign coding utility to **94.44%** (`17/18`) while keeping read-only external Q&A intact (overall episode McNemar `p = 0.062500`, benign utility `p = 0.062500`, `b = 5`, `c = 0`).
- **Uncached 20-Note Live Write Latency**: `LayaAdmissionGate` runs locally in **400.837 ms** mean (**400.634 ms** median, **480.260 ms** p95) vs **550.466 ms** mean (**511.739 ms** median, **1169.238 ms** p95) for uncached cloud `LLMAdmissionGate` (**1.37x** faster mean, **2.43x** faster p95, zero cloud API calls), alongside `0.365 ms` mean for `LexicalAdmissionGate` and `0.944 ms` mean for `StockUnprotectedGate`.

## Repository Layout

- `pgmem/store.py`: SQLite memory ledger with `HMAC-SHA256` row verification, word+bigram TF-IDF cosine similarity, authority-dominance clone shadowing, quarantine, and BM25 retrieval.
- `pgmem/gates.py`: `StockUnprotectedGate`, `LLMAdmissionGate`, `LexicalAdmissionGate`, `LayaAdmissionGate` (`convaiinnovations/laya` 421M), and `ToolInterceptor` (`none`, `naive_biba_min`, `clean_room`).
- `pgmem/llm.py`: Thread-safe stdlib `urllib.request` Gemini client with WAL SQLite prompt caching and rate-limit retries.
- `benchmarks/`: Deterministic generator (`build_suite.py`) producing `coding_memory_suite.json` (`N = 36`) and `recall_slice.json` (`N = 30`).
- `run_mvp.py`: End-to-end 6-architecture evaluation runner writing `results/mvp_summary.json`.
- `demo/simulator.html`: Self-contained interactive 3-step science-fair simulator powered by live episode logs from `results/mvp_summary.json`.
- `tests/test_pgmem_unit.py`: Unit tests covering HMAC tamper rejection, authority-dominance clone shadowing, Laya 4-question gating, and Clean-Room non-interference.

## Reproducing From Scratch

```bash
pytest -q
python benchmarks/build_suite.py
python run_mvp.py
```
