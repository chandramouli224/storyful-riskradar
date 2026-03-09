from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from src.config import (
    CLUSTER_DISTANCE_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    JUNK_LABEL_TERMS,
    JUNK_TEXT_PATTERNS,
    MIN_USABLE_NARRATIVE_SIZE,
)


def flag_weak_signal_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weak-signal heuristics to posts.

    These are not used to remove posts by default, but are useful for:
    - diagnostics
    - confidence
    - possible future filtering
    """
    out = posts_df.copy()

    out["has_url"] = out["text"].str.contains(r"http\S+|www\.\S+", regex=True, na=False)
    out["n_mentions"] = out["text"].str.count(r"@\w+")
    out["n_hashtags"] = out["text"].str.count(r"#\w+")
    out["word_count"] = out["text_clean"].str.split().str.len()
    out["resolved_group_count"] = out["n_resolved_groups"]

    out["very_short_text"] = out["word_count"] < 6
    out["too_many_entities"] = out["resolved_group_count"] >= 4
    out["too_many_mentions"] = out["n_mentions"] >= 3
    out["too_many_hashtags"] = out["n_hashtags"] >= 6

    out["weak_signal_post"] = (
        out["very_short_text"]
        | out["too_many_entities"]
        | out["too_many_mentions"]
        | out["too_many_hashtags"]
    )

    return out


def _cluster_single_theme(
    subdf: pd.DataFrame,
    embedding_model: SentenceTransformer,
    distance_threshold: float,
) -> pd.DataFrame:
    """
    Cluster posts inside a single theme bucket.
    """
    subdf = subdf.copy().reset_index(drop=True)

    if len(subdf) == 1:
        subdf["theme_cluster_id"] = 0
        return subdf

    embeddings = embedding_model.encode(
        subdf["text_clean"].tolist(),
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    dist = cosine_distances(embeddings)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )

    labels = clusterer.fit_predict(dist)
    subdf["theme_cluster_id"] = labels

    return subdf


def cluster_within_themes(
    posts_df: pd.DataFrame,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Cluster theme-qualified posts within each primary theme.

    Required columns:
        post_id
        text_clean
        primary_theme
    """
    required_cols = {"post_id", "text_clean", "primary_theme"}
    missing = required_cols - set(posts_df.columns)
    if missing:
        raise ValueError(f"cluster_within_themes missing required columns: {missing}")

    clustered_parts: list[pd.DataFrame] = []
    embedding_model = SentenceTransformer(embedding_model_name)

    for theme, subdf in posts_df.groupby("primary_theme"):
        if theme == "misc":
            continue

        clustered = _cluster_single_theme(
            subdf=subdf,
            embedding_model=embedding_model,
            distance_threshold=distance_threshold,
        )
        clustered["narrative_id"] = clustered["theme_cluster_id"].apply(
            lambda x: f"{theme}_{x}"
        )
        clustered_parts.append(clustered)

    if not clustered_parts:
        return pd.DataFrame(
            columns=list(posts_df.columns) + ["theme_cluster_id", "narrative_id"]
        )

    return pd.concat(clustered_parts, ignore_index=True)


def _build_tfidf_labels(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build TF-IDF keyword labels for narratives.
    """
    if clustered_df.empty:
        return pd.DataFrame(
            columns=["narrative_id", "primary_theme", "size", "label_terms"]
        )

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=2000,
    )

    X_tfidf = tfidf_vectorizer.fit_transform(clustered_df["text_clean"])
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    label_rows: list[dict[str, Any]] = []

    for narrative_id, sub in clustered_df.groupby("narrative_id"):
        idx = sub.index.tolist()
        cluster_matrix = X_tfidf[idx]
        mean_scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:20]
        raw_terms = feature_names[top_idx].tolist()

        cleaned_terms = clean_top_terms(raw_terms, top_n=6)
        if not cleaned_terms:
            cleaned_terms = raw_terms[:6]

        label_rows.append(
            {
                "narrative_id": narrative_id,
                "primary_theme": sub["primary_theme"].iloc[0],
                "size": len(sub),
                "label_terms": ", ".join(cleaned_terms),
            }
        )

    return (
        pd.DataFrame(label_rows)
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )


def clean_top_terms(raw_terms: list[str], top_n: int = 6) -> list[str]:
    """
    Remove junk terms from candidate TF-IDF labels.
    """
    cleaned: list[str] = []

    for term in raw_terms:
        t = term.strip().lower()
        if t in JUNK_LABEL_TERMS:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        if len(t) <= 2:
            continue

        cleaned.append(term)
        if len(cleaned) >= top_n:
            break

    return cleaned


def build_narrative_labels(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Public wrapper to build narrative labels.
    """
    return _build_tfidf_labels(clustered_df)


def narrative_is_probably_junk(label_terms: str, size: int) -> bool:
    """
    Heuristic junk detector for narrative labels.
    """
    text = str(label_terms).lower()

    junk_hits = sum(
        1
        for term in JUNK_LABEL_TERMS
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text)
    )
    pattern_hits = sum(1 for pattern in JUNK_TEXT_PATTERNS if re.search(pattern, text))

    if size == 1:
        return True
    if junk_hits >= 2:
        return True
    if pattern_hits >= 1:
        return True

    return False


def mark_probably_junk_narratives(narrative_labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean junk flag to narrative labels.
    """
    out = narrative_labels_df.copy()
    out["is_probably_junk"] = out.apply(
        lambda row: narrative_is_probably_junk(
            label_terms=str(row["label_terms"]),
            size=int(row["size"]),
        ),
        axis=1,
    )
    return out


def filter_usable_narratives(
    narrative_labels_df: pd.DataFrame,
    min_size: int = MIN_USABLE_NARRATIVE_SIZE,
) -> pd.DataFrame:
    """
    Keep only usable narratives for scoring.
    """
    required_cols = {"narrative_id", "size", "is_probably_junk"}
    missing = required_cols - set(narrative_labels_df.columns)
    if missing:
        raise ValueError(
            f"filter_usable_narratives missing required columns: {missing}"
        )

    usable = narrative_labels_df[
        (~narrative_labels_df["is_probably_junk"])
        & (narrative_labels_df["size"] >= min_size)
    ].copy()

    return usable.reset_index(drop=True)


def attach_only_usable_posts(
    clustered_posts: pd.DataFrame,
    usable_narratives_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only posts belonging to usable narratives.
    """
    usable_ids = set(usable_narratives_df["narrative_id"].tolist())
    out = clustered_posts[clustered_posts["narrative_id"].isin(usable_ids)].copy()
    return out.reset_index(drop=True)
