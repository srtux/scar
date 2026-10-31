# MemoryGuard: What AI Agents Should Remember and Trust

Coding assistants save persistent notes across sessions to remember project test flags, directory layouts, and personal workflow preferences. Saving raw session logs causes two problems: casual small talk clutters retrieved context, and untrusted web pages or GitHub issues can plant fake conventions that hijack tool calls in a later session.

MemoryGuard (`memory_guard`) solves both problems in under 2,000 lines of Python without blocking normal coding work or calling a cloud LLM on memory writes.

## Project Goals

MemoryGuard is built around four concrete goals:

1. Stop both imperative prompt injections and calm convention-poisoning attacks without blocking legitimate commands when ordinary external notes sit in memory.
2. Filter everyday greetings and filler turns while preserving personal preference recall across multi-session chats.
3. Score candidate notes locally on CPU with a 421M-parameter decision model (`convaiinnovations/laya`) so memory writes make zero cloud API calls.
4. Keep the codebase, datasets, signed SQLite store, test suite, and SQLite prompt-replay cache self-contained in a standard Python virtual environment.

## Quickstart

You need Python 3.10 or newer and ~1.7 GB of free disk space on the first run for the local `Laya-421M` weights from Hugging Face.

```bash
git clone https://github.com/srtux/memoryguard.git
cd memoryguard

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

On Windows PowerShell, activate the virtual environment with `.venv\Scripts\Activate.ps1`. Open `.env` and set `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey), or set `GOOGLE_CLOUD_PROJECT` to authenticate through Google Cloud Vertex AI. You can also set `GEMINI_MODEL` in `.env` to override the default `gemini-3.5-flash-lite` model.

Run the offline test suite first (no credentials or cloud network access required), then validate the benchmark suites and run the evaluation:

```bash
python -m pytest -v
python benchmarks/build_suite.py
python run_mvp.py
```

The runner loads `.env` automatically and caches LLM responses in SQLite under `.cache/` so subsequent replays finish in seconds with zero API calls.

## Experiment

MemoryGuard separates write-time admission from action-time verification in three stages:

1. `LayaAdmissionGate` runs `convaiinnovations/laya` (`typed-decisions`) on CPU across four probability heads (`utility`, `confidence`, `category`, and `danger`). It vetoes direct overrides first and drops low-utility chatter before storage.
2. `MemoryStore` tags notes as `USER` (`3`), `REPO` (`2`), or `EXTERNAL` (`1`), signs rows and active flags with `HMAC-SHA256`, indexes text with SQLite `FTS5` BM25, and drops untrusted clones of trusted rules as `shadowed`.
3. Before running shell commands, file writes, HTTP posts, sensitive file reads (`.env`, SSH credentials), or URL-bearing answers, `ToolInterceptor` checks whether any `EXTERNAL` note entered retrieval. When untrusted notes are present, it replans with `USER` and `REPO` notes alone, quarantines untrusted notes on mismatch, and executes the clean trusted plan.

The evaluation compares MemoryGuard against five ablations (`Stock Unprotected`, `LLM Gate Only`, `Lexical Regex Gate Only`, `Laya-421M Gate Only`, and `Laya-421M + Blanket Red Lock`) across two paired benchmarks:

- The coding memory suite (`48` two-session episodes across `11` software domains) mixes `USER`, `REPO`, and `EXTERNAL` sources across `24` unattacked tasks and `24` cross-session attacks (`12` direct overrides and `12` disguised conventions or split-turn fragments). An episode only passes when zero hijacking occurs and the developer's real coding task completes.
- The `PersonaMem-v1` recall slice (`30` dialogues of `40` turns) tests pure `USER` chat histories combining 25% personal facts with 75% small talk and zero code syntax. Pairing this slice with the coding suite stops brittle flag-matching regex filters from gaming evaluation: regex gating keeps `79.2%` benign coding utility by matching `--flags`, yet drops `PersonaMem-v1` recall to `33.3%`.

## Results

All six architectures were evaluated with `gemini-3.5-flash-lite` at temperature zero. Lower Attack Success Rate (ASR) and Benign Lock are better; higher utility, episode success, recall, and storage savings are better:

| Architecture | Overall ASR | Sneaky ASR | Benign Lock | Benign Utility | Attacked Utility | Overall Success | PersonaMem Recall | Storage Saved | Write Latency Mean (p95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Unprotected | 50.0% | 83.3% | 0.0% | 100.0% | 50.0% | 75.0% | 73.3% | 0.0% | 0.8 ms (1.4 ms) |
| LLM Gate Only (A-MAC style) | 41.7% | 83.3% | 0.0% | 100.0% | 58.3% | 79.2% | 60.0% | 51.5% | 555 ms (1,550 ms) |
| Lexical Regex Gate Only | 29.2% | 58.3% | 0.0% | 79.2% | 62.5% | 70.8% | 33.3% | 85.9% | 0.2 ms (0.3 ms) |
| Laya-421M Gate Only | 20.8% | 41.7% | 0.0% | 100.0% | 79.2% | 89.6% | 76.7% | 23.0% | 726 ms (893 ms) |
| Laya-421M + Blanket Red Lock | 0.0% | 0.0% | 33.3% | 66.7% | 70.8% | 68.8% | 76.7% | 23.0% | 726 ms (893 ms) |
| **MemoryGuard** | **0.0%** | **0.0%** | **0.0%** | **100.0%** | **100.0%** | **100.0%** | **76.7%** | **23.0%** | **726 ms (893 ms)** |

Paired exact two-sided McNemar tests and latency takeaways:

- MemoryGuard cuts overall ASR from `50.0%` (`12/24`) on Stock Unprotected and `41.7%` (`10/24`) on LLM Gate Only down to `0.0%` (`0/24`, McNemar p-values `0.0005` and `0.0020`).
- Blanket Red Lock blocks `33.3%` of unattacked tasks (`88.9%` on tainted privileged tasks). Clean-Room replanning eliminates false locks (`0.0%`, p-value `0.0078`) and recovers all `48` episodes to the clean trusted action (`100.0%` vs `68.8%` overall success, p-value `0.00006`).
- `Laya-421M` lifts `PersonaMem-v1` recall from `73.3%` (`22/30`) to `76.7%` (`23/30`) while cutting stored words by `23.0%`. It outperforms Lexical Regex Gate Only across overall coding success (p-value `0.0001`) and conversational recall (`76.7%` vs `33.3%`).
- On the uncached 20-note write benchmark, local CPU inference with `Laya-421M` runs with zero cloud API calls and a 1.7x tighter p95 tail (`893 ms`) than uncached cloud LLM gating (`1,550 ms`).

## Documentation

Ground-up documentation for the project lives under `docs/`:

| Guide | Topics Covered |
| :--- | :--- |
| [Overview and Threat Model](docs/overview.md) | Provenance tiers, cross-session attack taxonomy, single-layer limits, and prior work |
| [System Architecture](docs/architecture.md) | Pipeline dataflow, `Laya-421M` two-stage equations, signed SQLite + `FTS5`, and Clean-Room equivalence |
| [Benchmark Datasets](docs/datasets.md) | Why both suites are paired, 48-episode cohort design, causal leak stripping, and `PersonaMem` `v1` vs `v2`/`v3` |
| [Evaluation and Reproducibility](docs/evaluation.md) | Six-arm ablations, metric definitions, exact McNemar tables, write latency, offline replay, and pytest suite |
