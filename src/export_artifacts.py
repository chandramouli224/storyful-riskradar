from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_entity_overview(
    posts_resolved: pd.DataFrame,
    final_narratives: pd.DataFrame,
    entity_name: str,
) -> pd.DataFrame:
    """
    Build one-row entity overview for Streamlit.
    """
    matched_mask = posts_resolved["resolved_entities"].apply(
        lambda ents: any(e.get("group_name") == entity_name for e in ents)
    )

    matched_posts = int(matched_mask.sum())
    pct_posts_matched = (
        round(float(matched_mask.mean() * 100), 2) if len(posts_resolved) else 0.0
    )

    n_narratives = (
        int(final_narratives["narrative_id"].nunique())
        if not final_narratives.empty
        else 0
    )
    avg_confidence = (
        round(float(final_narratives["confidence_score"].mean()), 1)
        if not final_narratives.empty
        else 0.0
    )
    n_low_support_narratives = (
        int((final_narratives["n_posts"] < 3).sum())
        if not final_narratives.empty
        else 0
    )

    return pd.DataFrame(
        [
            {
                "entity_name": entity_name,
                "entity_id_display": entity_name.lower().replace(" ", "_"),
                "matched_posts": matched_posts,
                "pct_posts_matched": pct_posts_matched,
                "n_narratives": n_narratives,
                "avg_confidence": avg_confidence,
                "n_low_support_narratives": n_low_support_narratives,
                "notes": "Rules-first grouped entity resolution; theme-first narrative formation.",
            }
        ]
    )


def prepare_narratives_export(
    final_narratives: pd.DataFrame, entity_name: str
) -> pd.DataFrame:
    """
    Prepare scored narratives for app consumption.
    """
    out = final_narratives.copy()

    out = out.rename(
        columns={
            "primary_theme": "theme",
        }
    )

    out["entity_name"] = entity_name
    out["driver_tags"] = out["driver_tags"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else str(x)
    )

    ordered_cols = [
        "entity_name",
        "narrative_id",
        "theme",
        "label",
        "n_posts",
        "n_unique_authors",
        "risk_score",
        "confidence_score",
        "driver_tags",
        "volume_norm",
        "velocity_norm",
        "engagement_norm",
        "author_norm",
        "language_risk_norm",
        "theme_prior_norm",
    ]

    existing_cols = [c for c in ordered_cols if c in out.columns]
    return out[existing_cols].copy()


def prepare_evidence_export(
    evidence_df: pd.DataFrame, entity_name: str
) -> pd.DataFrame:
    """
    Prepare evidence posts for app consumption.
    """
    out = evidence_df.copy()
    out["entity_name"] = entity_name

    ordered_cols = [
        "entity_name",
        "narrative_id",
        "evidence_rank",
        "post_id",
        "created_at",
        "platform",
        "text",
        "weighted_engagement",
        "risk_term_hits",
        "followers",
    ]

    existing_cols = [c for c in ordered_cols if c in out.columns]
    return out[existing_cols].copy()


def prepare_posts_export(
    narrative_posts_scored: pd.DataFrame, entity_name: str
) -> pd.DataFrame:
    """
    Prepare narrative member posts for app/debug consumption.
    """
    out = narrative_posts_scored.copy()
    out = out.rename(columns={"primary_theme": "theme"})
    out["entity_name"] = entity_name

    ordered_cols = [
        "entity_name",
        "narrative_id",
        "theme",
        "post_id",
        "author_id",
        "created_at",
        "platform",
        "text",
        "text_clean",
        "likes",
        "shares",
        "comments",
        "views",
        "weighted_engagement",
        "risk_term_hits",
    ]

    existing_cols = [c for c in ordered_cols if c in out.columns]
    return out[existing_cols].copy()


def write_artifacts(
    output_dir: str | Path,
    entity_overview_df: pd.DataFrame,
    narratives_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    posts_df: pd.DataFrame,
) -> None:
    """
    Write final CSV artifacts to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_overview_df.to_csv(output_dir / "entity_overview.csv", index=False)
    narratives_df.to_csv(output_dir / "narratives.csv", index=False)
    evidence_df.to_csv(output_dir / "narrative_evidence.csv", index=False)
    posts_df.to_csv(output_dir / "narrative_posts.csv", index=False)
