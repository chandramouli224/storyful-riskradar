from __future__ import annotations

import pandas as pd


def select_evidence_posts(narrative_posts_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Select top evidence posts per narrative using a simple weighted evidence score.

    Output columns:
        narrative_id
        evidence_rank
        post_id
        created_at
        platform
        text
        weighted_engagement
        risk_term_hits
        followers
    """
    evidence_rows = []

    for narrative_id, sub in narrative_posts_scored.groupby("narrative_id"):
        sub = sub.copy()

        sub["evidence_score"] = (
            0.45 * sub["log_engagement"].fillna(0)
            + 0.30 * sub["risk_term_hits"].fillna(0)
            + 0.15 * sub["is_last_7d"].fillna(0)
            + 0.10 * sub["author_influence_proxy"].fillna(0)
        )

        sub = sub.sort_values("evidence_score", ascending=False)

        # Deduplicate by narrative + post_id first
        if "post_id" in sub.columns:
            sub = sub.drop_duplicates(subset=["post_id"], keep="first")

        # Safety fallback: deduplicate repeated text within a narrative
        sub = sub.drop_duplicates(subset=["text"], keep="first")

        top_sub = sub.head(3)

        for rank, (_, row) in enumerate(top_sub.iterrows(), start=1):
            evidence_rows.append(
                {
                    "narrative_id": narrative_id,
                    "evidence_rank": rank,
                    "post_id": row["post_id"],
                    "created_at": row["created_at"],
                    "platform": row["platform"],
                    "text": row["text"],
                    "weighted_engagement": row["weighted_engagement"],
                    "risk_term_hits": row["risk_term_hits"],
                    "followers": row["followers"],
                }
            )

    evidence_df = pd.DataFrame(evidence_rows)
    return evidence_df


def deduplicate_evidence(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Final deduplication / reranking pass for evidence output.
    """
    if evidence_df.empty:
        return evidence_df.copy()

    out = evidence_df.copy()
    out = out.sort_values(["narrative_id", "evidence_rank"], ascending=[True, True])

    if "post_id" in out.columns:
        out = out.drop_duplicates(subset=["narrative_id", "post_id"], keep="first")

    out = out.drop_duplicates(subset=["narrative_id", "text"], keep="first")
    out["evidence_rank"] = out.groupby("narrative_id").cumcount() + 1
    out = out[out["evidence_rank"] <= 3].copy()

    return out.reset_index(drop=True)
