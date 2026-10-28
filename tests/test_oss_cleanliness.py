"""Repository hygiene: no internal references, no personal paths, no leaked keys, and no Unicode dashes."""

import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKIP_DIRS = {".git", ".cache", "__pycache__", ".pytest_cache", "data", ".venv"}
TEXT_EXTS = (".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".html", ".svg")
FORBIDDEN = [r"\bgoogle3\b", r"/google/src/", r"/google/bin/", r"\bcorp\.google\.com\b", r"\bgoogleplex\.com\b",
             r"/usr/local/google/home", r"\.bashrc", r"\bsummitt-gcp\b", r"\bsummitt-cloud\b",
             r"AIza[0-9A-Za-z_\-]{30,}", r"sk-[0-9A-Za-z]{30,}"]


def _text_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(TEXT_EXTS) and name != "test_oss_cleanliness.py":
                path = os.path.join(root, name)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    yield os.path.relpath(path, REPO), f.read()


def test_no_internal_references_personal_paths_or_keys():
    patterns = [re.compile(p) for p in FORBIDDEN]
    for rel, text in _text_files():
        for pat in patterns:
            assert pat.search(text) is None, f"`{pat.pattern}` found in {rel}"


def test_no_unicode_em_or_en_dashes():
    for rel, text in _text_files():
        assert "\u2014" not in text and "\u2013" not in text, f"Unicode dash in {rel}"
