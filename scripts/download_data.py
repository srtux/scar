"""Download public MIT-licensed evaluation corpora (`tau2-bench` and `OpenHands/openhands-feedback`)."""

from __future__ import annotations

import json
import os
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAU2_DIR = os.path.join(REPO_ROOT, "data", "tau2")
OH_DIR = os.path.join(REPO_ROOT, "data", "openhands")

TAU2_RAW = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/results/final"
TAU2_FILES = [
    "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json",
    "gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json",
    "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json",
    "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json",
]


def download_all() -> None:
    os.makedirs(TAU2_DIR, exist_ok=True)
    os.makedirs(OH_DIR, exist_ok=True)

    for fname in TAU2_FILES:
        dest = os.path.join(TAU2_DIR, fname)
        if not os.path.exists(dest):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(f"{TAU2_RAW}/{fname}", dest)
        print(f"Verified {fname} ({os.path.getsize(dest) / (1024 * 1024):.2f} MB)")

    oh_jsonl = os.path.join(OH_DIR, "openhands_feedback_sessions.jsonl")
    if os.path.exists(oh_jsonl):
        with open(oh_jsonl) as f:
            n_rows = sum(1 for line in f if line.strip())
        print(f"Verified openhands_feedback_sessions.jsonl ({n_rows} sessions)")
    else:
        print(
            "openhands_feedback_sessions.jsonl missing; fetch OpenHands/openhands-feedback "
            "from HuggingFace (`data/train-00000-of-00001.parquet`) to regenerate."
        )


if __name__ == "__main__":
    download_all()
