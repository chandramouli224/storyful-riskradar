from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_posts(path: str | Path) -> pd.DataFrame:
    """Load posts.jsonl and standardize core columns."""
    posts = pd.read_json(path, lines=True)
    posts = posts.copy()

    # Required fields cleanup
    if "post_id" in posts.columns:
        posts["post_id"] = posts["post_id"].astype(str)

    if "author_id" in posts.columns:
        posts["author_id"] = posts["author_id"].astype(str)

    if "created_at" in posts.columns:
        posts["created_at"] = pd.to_datetime(
            posts["created_at"], errors="coerce", utc=True
        )

    if "text" not in posts.columns:
        posts["text"] = ""
    posts["text"] = posts["text"].fillna("").astype(str)

    if "language" in posts.columns:
        posts["language"] = (
            posts["language"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "unknown": np.nan, "None": np.nan})
            .str.lower()
        )

    if "platform" in posts.columns:
        posts["platform"] = posts["platform"].astype(str).str.strip().str.lower()

    for col in ["likes", "shares", "comments", "views"]:
        if col in posts.columns:
            posts[col] = pd.to_numeric(posts[col], errors="coerce")
        else:
            posts[col] = 0.0

    return posts


def load_authors(path: str | Path) -> pd.DataFrame:
    """Load authors.csv and standardize fields."""
    authors = pd.read_csv(path)
    authors = authors.copy()

    if "author_id" in authors.columns:
        authors["author_id"] = authors["author_id"].astype(str)

    for col in ["followers", "account_age_days"]:
        if col in authors.columns:
            authors[col] = pd.to_numeric(authors[col], errors="coerce")

    if "verified" in authors.columns:
        authors["verified"] = (
            authors["verified"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
    else:
        authors["verified"] = 0

    for col in ["handle", "display_name"]:
        if col not in authors.columns:
            authors[col] = np.nan

    return authors


def load_entities(path: str | Path) -> pd.DataFrame:
    """Load entities_seed.csv and standardize fields."""
    entities = pd.read_csv(path)
    entities = entities.copy()

    for col in ["entity_id", "canonical_name", "entity_type"]:
        if col in entities.columns:
            entities[col] = entities[col].astype(str).str.strip()

    if "aliases" in entities.columns:
        entities["aliases"] = entities["aliases"].fillna("").astype(str)
    else:
        entities["aliases"] = ""

    return entities


def load_all(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all raw challenge inputs from a directory."""
    data_dir = Path(data_dir)

    posts = load_posts(data_dir / "posts.jsonl")
    authors = load_authors(data_dir / "authors.csv")
    entities = load_entities(data_dir / "entities_seed.csv")

    return posts, authors, entities
