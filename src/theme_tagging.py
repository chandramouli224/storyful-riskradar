from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import THEME_LEXICON, THEME_PRIORITY
from src.utils import count_regex_hits


def count_theme_hits(
    text: str, lexicon: dict[str, list[str]] | None = None
) -> dict[str, dict[str, Any]]:
    """
    Count per-theme keyword hits for a post.

    Returns:
        {
            "theme_name": {
                "count": int,
                "matched_terms": list[str]
            },
            ...
        }
    """
    lexicon = lexicon or THEME_LEXICON
    hits: dict[str, dict[str, Any]] = {}

    for theme, keywords in lexicon.items():
        count, matched_terms = count_regex_hits(text, keywords)
        hits[theme] = {
            "count": count,
            "matched_terms": matched_terms,
        }

    return hits


def choose_primary_theme(
    theme_hits: dict[str, dict[str, Any]],
) -> tuple[str, int, list[str]]:
    """
    Choose a primary theme using:
    1. highest keyword count
    2. theme priority as tie-breaker

    Returns:
        (primary_theme, primary_theme_score, matched_themes)
    """
    best_theme = "misc"
    best_count = 0
    best_priority = -1
    matched_themes: list[str] = []

    for theme, info in theme_hits.items():
        count = int(info["count"])

        if count > 0:
            matched_themes.append(theme)

        if count > best_count:
            best_theme = theme
            best_count = count
            best_priority = THEME_PRIORITY.get(theme, 0)
        elif count == best_count and count > 0:
            current_priority = THEME_PRIORITY.get(theme, 0)
            if current_priority > best_priority:
                best_theme = theme
                best_priority = current_priority

    return best_theme, best_count, matched_themes


def assign_primary_theme(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign theme hits and a primary theme to each post.

    Required columns:
        text_clean

    Output columns added:
        theme_hits
        matched_themes
        primary_theme
        primary_theme_score
    """
    themed_posts = posts_df.copy()

    if "text_clean" not in themed_posts.columns:
        raise ValueError("assign_primary_theme requires a 'text_clean' column.")

    rows: list[dict[str, Any]] = []

    for _, row in themed_posts.iterrows():
        theme_hits = count_theme_hits(str(row["text_clean"]))
        primary_theme, primary_theme_score, matched_themes = choose_primary_theme(
            theme_hits
        )

        rows.append(
            {
                "post_id": row["post_id"],
                "theme_hits": theme_hits,
                "matched_themes": matched_themes,
                "primary_theme": primary_theme,
                "primary_theme_score": primary_theme_score,
            }
        )

    theme_df = pd.DataFrame(rows)
    themed_posts = themed_posts.merge(theme_df, on="post_id", how="left")

    return themed_posts


def filter_theme_qualified_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep posts that have a non-misc primary theme and at least one theme hit.
    """
    required_cols = {"primary_theme", "primary_theme_score"}
    missing = required_cols - set(posts_df.columns)
    if missing:
        raise ValueError(
            f"filter_theme_qualified_posts missing required columns: {missing}"
        )

    qualified = posts_df[
        (posts_df["primary_theme"] != "misc") & (posts_df["primary_theme_score"] >= 1)
    ].copy()

    return qualified.reset_index(drop=True)


def build_theme_distribution(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple theme distribution table.

    Output columns:
        primary_theme
        count
        pct
    """
    if "primary_theme" not in posts_df.columns:
        raise ValueError("build_theme_distribution requires a 'primary_theme' column.")

    theme_dist = (
        posts_df["primary_theme"]
        .value_counts()
        .rename_axis("primary_theme")
        .reset_index(name="count")
    )

    total = len(posts_df)
    if total == 0:
        theme_dist["pct"] = 0.0
    else:
        theme_dist["pct"] = (theme_dist["count"] / total * 100).round(2)

    return theme_dist


def add_text_clean(posts_df: pd.DataFrame, clean_fn) -> pd.DataFrame:
    """
    Convenience helper to add text_clean if not already present.
    Expects clean_fn to be a callable like utils.clean_text_for_clustering.
    """
    out = posts_df.copy()

    if "text" not in out.columns:
        raise ValueError("add_text_clean requires a 'text' column.")

    out["text_clean"] = out["text"].fillna("").astype(str).apply(clean_fn)
    return out
