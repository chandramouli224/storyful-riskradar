from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CONFIDENCE_WEIGHTS,
    EMBEDDING_MODEL_NAME,
    RISK_LEXICON,
    SCORING_WEIGHTS,
    THEME_PRIOR,
)
from src.utils import count_regex_hits, minmax, safe_log1p


def prepare_post_level_features(
    narrative_posts: pd.DataFrame,
    authors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join author metadata and compute post-level features used for scoring.
    """
    posts = narrative_posts.copy()
    auth = authors.copy()

    posts["author_id"] = posts["author_id"].astype(str)
    auth["author_id"] = auth["author_id"].astype(str)

    join_cols = [
        c
        for c in ["author_id", "followers", "verified", "handle", "display_name"]
        if c in auth.columns
    ]
    posts = posts.merge(auth[join_cols], on="author_id", how="left")

    for col in ["likes", "shares", "comments", "views", "followers"]:
        if col in posts.columns:
            posts[col] = pd.to_numeric(posts[col], errors="coerce")
        else:
            posts[col] = 0.0

    if "verified" in posts.columns:
        posts["verified"] = (
            posts["verified"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
    else:
        posts["verified"] = 0

    # Weighted engagement: simple and explainable
    posts["weighted_engagement"] = (
        posts["likes"].fillna(0) * 1.0
        + posts["comments"].fillna(0) * 2.0
        + posts["shares"].fillna(0) * 3.0
        + posts["views"].fillna(0) * 0.001
    )

    posts["log_engagement"] = safe_log1p(posts["weighted_engagement"])
    posts["log_followers"] = safe_log1p(posts["followers"])
    posts["author_influence_proxy"] = (
        posts["log_followers"] + posts["verified"].fillna(0) * 1.0
    )

    entity_max_ts = posts["created_at"].max()
    posts["days_from_entity_max"] = (
        entity_max_ts - posts["created_at"]
    ).dt.total_seconds() / 86400.0

    posts["is_last_3d"] = (posts["days_from_entity_max"] <= 3).astype(int)
    posts["is_last_7d"] = (posts["days_from_entity_max"] <= 7).astype(int)

    return posts


def count_risk_terms(
    text: str, lexicon: dict[str, list[str]] | None = None
) -> tuple[int, list[str]]:
    """
    Count risk-language terms in a text span.
    """
    lexicon = lexicon or RISK_LEXICON
    total_hits = 0
    matched_terms: list[str] = []

    for _, keywords in lexicon.items():
        count, matched = count_regex_hits(text, keywords)
        total_hits += count
        matched_terms.extend(matched)

    return total_hits, sorted(set(matched_terms))


def add_risk_term_features(narrative_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Add post-level risk term hits and matched term lists.
    """
    posts = narrative_posts.copy()
    rows: list[dict[str, Any]] = []

    for _, row in posts.iterrows():
        n_hits, matched_terms = count_risk_terms(str(row["text_clean"]))
        rows.append(
            {
                "post_id": row["post_id"],
                "risk_term_hits": n_hits,
                "risk_terms_matched": matched_terms,
            }
        )

    risk_df = pd.DataFrame(rows)
    posts = posts.merge(risk_df, on="post_id", how="left")
    posts["risk_term_hits"] = posts["risk_term_hits"].fillna(0).astype(int)

    return posts


def aggregate_narrative_features(narrative_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate post-level features to narrative-level features.
    """
    rows: list[dict[str, Any]] = []

    for narrative_id, sub in narrative_posts.groupby("narrative_id"):
        sub = sub.sort_values("created_at")
        theme = sub["primary_theme"].iloc[0]

        n_posts = len(sub)
        n_unique_authors = sub["author_id"].nunique()
        mean_log_engagement = float(sub["log_engagement"].mean())
        mean_author_influence = float(sub["author_influence_proxy"].mean())
        total_risk_hits = int(sub["risk_term_hits"].sum())
        pct_last_3d = float(sub["is_last_3d"].mean())
        pct_last_7d = float(sub["is_last_7d"].mean())

        posts_per_day = sub.groupby(sub["created_at"].dt.date).size()
        max_posts_one_day = int(posts_per_day.max()) if len(posts_per_day) else 1

        follower_cov = (
            float(sub["followers"].notna().mean())
            if "followers" in sub.columns
            else 0.0
        )
        handle_cov = (
            float(sub["handle"].notna().mean()) if "handle" in sub.columns else 0.0
        )

        rows.append(
            {
                "narrative_id": narrative_id,
                "primary_theme": theme,
                "n_posts": n_posts,
                "n_unique_authors": n_unique_authors,
                "mean_log_engagement": mean_log_engagement,
                "mean_author_influence": mean_author_influence,
                "total_risk_hits": total_risk_hits,
                "pct_last_3d": pct_last_3d,
                "pct_last_7d": pct_last_7d,
                "max_posts_one_day": max_posts_one_day,
                "follower_cov": follower_cov,
                "handle_cov": handle_cov,
                "theme_prior": THEME_PRIOR.get(theme, 0.50),
            }
        )

    return pd.DataFrame(rows)


def compute_semantic_coherence(
    narrative_posts: pd.DataFrame,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> pd.DataFrame:
    """
    Compute mean centroid similarity for each narrative.
    """
    model = SentenceTransformer(embedding_model_name)
    rows: list[dict[str, Any]] = []

    for narrative_id, sub in narrative_posts.groupby("narrative_id"):
        texts = sub["text_clean"].tolist()

        if len(texts) == 1:
            rows.append(
                {
                    "narrative_id": narrative_id,
                    "mean_centroid_similarity": 0.50,
                }
            )
            continue

        emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        centroid = emb.mean(axis=0, keepdims=True)
        sims = cosine_similarity(emb, centroid).ravel()

        rows.append(
            {
                "narrative_id": narrative_id,
                "mean_centroid_similarity": float(np.mean(sims)),
            }
        )

    return pd.DataFrame(rows)


def score_narratives(narrative_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized components, risk score, and confidence score.
    """
    scored = narrative_features.copy()

    scored["volume_norm"] = minmax(np.log1p(scored["n_posts"]))

    velocity_base = (
        0.6 * scored["pct_last_3d"]
        + 0.4 * scored["pct_last_7d"]
        + 0.2 * minmax(scored["max_posts_one_day"])
    )
    scored["velocity_norm"] = minmax(velocity_base)

    scored["engagement_norm"] = minmax(scored["mean_log_engagement"])
    scored["author_norm"] = minmax(scored["mean_author_influence"])
    scored["language_risk_norm"] = minmax(np.log1p(scored["total_risk_hits"]))
    scored["theme_prior_norm"] = scored["theme_prior"]

    scored["risk_score"] = 100 * (
        SCORING_WEIGHTS["volume"] * scored["volume_norm"]
        + SCORING_WEIGHTS["velocity"] * scored["velocity_norm"]
        + SCORING_WEIGHTS["engagement"] * scored["engagement_norm"]
        + SCORING_WEIGHTS["author"] * scored["author_norm"]
        + SCORING_WEIGHTS["language_risk"] * scored["language_risk_norm"]
        + SCORING_WEIGHTS["theme_prior"] * scored["theme_prior_norm"]
    )

    scored["confidence_score"] = 100 * (
        CONFIDENCE_WEIGHTS["coherence"] * minmax(scored["mean_centroid_similarity"])
        + CONFIDENCE_WEIGHTS["size"] * minmax(np.log1p(scored["n_posts"]))
        + CONFIDENCE_WEIGHTS["follower_cov"] * scored["follower_cov"]
        + CONFIDENCE_WEIGHTS["handle_cov"] * scored["handle_cov"]
        + CONFIDENCE_WEIGHTS["author_diversity"]
        * minmax(np.log1p(scored["n_unique_authors"]))
    )

    scored["risk_score"] = scored["risk_score"].round(1)
    scored["confidence_score"] = scored["confidence_score"].round(1)

    return scored


def top_driver_tags(row: pd.Series) -> list[str]:
    """
    Pick top 3 driver tags from normalized scoring components.
    """
    components = {
        "volume_spike": float(row["volume_norm"]),
        "recent_velocity": float(row["velocity_norm"]),
        "high_engagement": float(row["engagement_norm"]),
        "high_influence_authors": float(row["author_norm"]),
        "risk_language": float(row["language_risk_norm"]),
        "theme_severity": float(row["theme_prior_norm"]),
    }

    ranked = sorted(components.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:3]]


def build_driver_tags(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add top 3 driver tags to scored narratives.
    """
    out = scored_df.copy()
    out["driver_tags"] = out.apply(top_driver_tags, axis=1)
    return out


def attach_labels_to_scores(
    scored_df: pd.DataFrame,
    narrative_labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach human-readable labels to scored narratives.
    """
    label_cols = [
        c
        for c in ["narrative_id", "label_terms", "size"]
        if c in narrative_labels_df.columns
    ]
    merged = scored_df.merge(
        narrative_labels_df[label_cols].rename(columns={"label_terms": "label"}),
        on="narrative_id",
        how="left",
    )
    return merged


def build_final_narratives(
    narrative_posts: pd.DataFrame,
    usable_narratives_df: pd.DataFrame,
    authors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end helper:
    - post-level features
    - risk term features
    - narrative-level aggregates
    - coherence
    - risk/confidence scoring
    - driver tags
    - labels attached

    Returns:
        final_narratives, narrative_posts_scored
    """
    posts_scored = prepare_post_level_features(narrative_posts, authors)
    posts_scored = add_risk_term_features(posts_scored)

    narrative_features = aggregate_narrative_features(posts_scored)
    coherence_df = compute_semantic_coherence(posts_scored)
    narrative_features = narrative_features.merge(
        coherence_df, on="narrative_id", how="left"
    )

    scored = score_narratives(narrative_features)
    scored = build_driver_tags(scored)
    final_narratives = attach_labels_to_scores(scored, usable_narratives_df)

    final_narratives = final_narratives.sort_values(
        ["risk_score", "confidence_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return final_narratives, posts_scored
