import re
from typing import Pattern

import numpy as np
import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize text for matching and grouping."""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s&-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_punctuation(text: str) -> str:
    """Light punctuation cleanup used for entity names."""
    text = str(text)
    text = text.replace(",", " ")
    text = re.sub(r"[.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_clustering(text: str) -> str:
    """Light cleaning for narrative clustering."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)  # keep hashtag token content, remove symbol
    text = re.sub(r"[^a-z0-9\s&\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compile_alias_pattern(alias: str) -> Pattern[str]:
    """Compile a whole-word style regex for entity alias matching."""
    alias = alias.strip()
    escaped = re.escape(alias.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def minmax(series: pd.Series) -> pd.Series:
    """Min-max scale a numeric series to [0, 1]."""
    s = pd.Series(series).astype(float)
    if len(s) == 0:
        return s
    if s.max() == s.min():
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def safe_log1p(series: pd.Series) -> pd.Series:
    """Numerically safe log1p transform."""
    s = pd.Series(series).fillna(0).clip(lower=0)
    return np.log1p(s)


def count_regex_hits(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Count whole-term keyword matches in text."""
    text_l = str(text).lower()
    matched_terms: list[str] = []

    for kw in keywords:
        pattern = rf"(?<!\w){re.escape(kw.lower())}(?!\w)"
        if re.search(pattern, text_l):
            matched_terms.append(kw)

    return len(matched_terms), sorted(set(matched_terms))
