"""Audit SCAR repository for OSS cleanliness, secret safety, and zero oracle leakage in src/scar."""

import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FORBIDDEN_INTERNAL_PATTERNS = [
    r"\bgoogle3\b",
    r"/google/src/",
    r"/google/bin/",
    r"\bcorp\.google\.com\b",
    r"\bgoogleplex\.com\b",
    r"\bsummitt-gcp\b",
    r"\bsummitt-cloud\b",
    r"AIza[0-9A-Za-z_\-]{30,}",
    r"sk-[0-9A-Za-z]{30,}",
]


def test_no_internal_references_or_leaked_keys():
    compiled = [re.compile(p) for p in FORBIDDEN_INTERNAL_PATTERNS]
    checked = 0
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", ".cache", "__pycache__", ".pytest_cache", "data")]
        for fname in files:
            if fname == "test_oss_cleanliness.py":
                continue
            if not fname.endswith((".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml")):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for pat in compiled:
                match = pat.search(content)
                assert match is None, f"Forbidden pattern `{pat.pattern}` found in {fpath}"
            checked += 1
    assert checked >= 8


def test_src_scar_never_reads_oracle_fields():
    src_dir = os.path.join(REPO_ROOT, "src", "scar")
    for fname in os.listdir(src_dir):
        if not fname.endswith(".py") or fname == "consolidator.py":
            continue
        fpath = os.path.join(src_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            code_lines = [
                line for line in f
                if not line.strip().startswith("#") and '"""' not in line
            ]
        joined = "\n".join(code_lines)
        assert '["reward_info"]' not in joined and '.get("reward_info")' not in joined, (
            f"Oracle field `reward_info` accessed inside {fname}"
        )
        assert '["nl_assertions"]' not in joined and '.get("nl_assertions")' not in joined, (
            f"Oracle field `nl_assertions` accessed inside {fname}"
        )
