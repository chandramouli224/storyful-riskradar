from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import CORP_SUFFIXES, MANUAL_GROUP_OVERRIDES
from src.utils import clean_punctuation, compile_alias_pattern, normalize_text


def strip_corp_suffixes(name: str) -> str:
    """
    Remove trailing corporate suffixes from an entity name.
    Example:
        'Pfizer Inc.' -> 'Pfizer'
        'AstraZeneca PLC' -> 'AstraZeneca'
    """
    name = clean_punctuation(name)
    tokens = name.split()

    while tokens and tokens[-1].lower() in CORP_SUFFIXES:
        tokens = tokens[:-1]

    cleaned = " ".join(tokens)
    if cleaned.endswith("&"):
        cleaned = cleaned[:-1].strip()

    return " ".join(cleaned.split())


def canonical_group_name(entity_id: str, canonical_name: str) -> str:
    """
    Build the analyst-facing grouped entity name.

    Manual overrides take precedence for known ambiguity cases.
    """
    if entity_id in MANUAL_GROUP_OVERRIDES:
        return MANUAL_GROUP_OVERRIDES[entity_id]

    return strip_corp_suffixes(canonical_name)


def generate_group_aliases(name: str) -> list[str]:
    """
    Generate lightweight alias variants for grouped entity matching.
    """
    name = str(name).strip()
    aliases: set[str] = set()

    aliases.add(name)
    aliases.add(clean_punctuation(name))
    aliases.add(name.replace("&", "and"))
    aliases.add(name.replace("-", " "))
    aliases.add(name.replace("-", ""))

    aliases = {alias.strip() for alias in aliases if alias and alias.strip()}
    aliases = {" ".join(alias.split()) for alias in aliases}

    return sorted(aliases)


def build_grouped_entities(entities: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw entity rows into analyst-facing grouped entities.

    Returns a dataframe with:
        entity_id
        canonical_name
        entity_type
        group_name
        group_name_norm
    """
    grouped_entities = entities.copy()

    grouped_entities["group_name"] = grouped_entities.apply(
        lambda row: canonical_group_name(
            entity_id=str(row["entity_id"]),
            canonical_name=str(row["canonical_name"]),
        ),
        axis=1,
    )
    grouped_entities["group_name_norm"] = grouped_entities["group_name"].apply(
        normalize_text
    )

    return grouped_entities


def build_group_alias_table(grouped_entities: pd.DataFrame) -> pd.DataFrame:
    """
    Build an alias table at the grouped-entity level.

    Output columns:
        group_name_norm
        group_name
        entity_type
        alias
        alias_norm
        candidate_entity_ids
        canonical_names
        pattern
    """
    grouped = (
        grouped_entities.groupby(
            ["group_name_norm", "group_name", "entity_type"], dropna=False
        )
        .agg(
            candidate_entity_ids=("entity_id", list),
            canonical_names=("canonical_name", list),
        )
        .reset_index()
    )

    grouped["aliases"] = grouped["group_name"].apply(generate_group_aliases)
    grouped["alias_count"] = grouped["aliases"].apply(len)

    alias_rows: list[dict[str, Any]] = []

    for _, row in grouped.iterrows():
        for alias in row["aliases"]:
            alias_rows.append(
                {
                    "group_name_norm": row["group_name_norm"],
                    "group_name": row["group_name"],
                    "entity_type": row["entity_type"],
                    "alias": alias,
                    "alias_norm": normalize_text(alias),
                    "candidate_entity_ids": tuple(row["candidate_entity_ids"]),
                    "canonical_names": tuple(row["canonical_names"]),
                }
            )

    group_alias_df = pd.DataFrame(alias_rows).drop_duplicates()
    group_alias_df["pattern"] = group_alias_df["alias"].apply(compile_alias_pattern)

    return group_alias_df


def _confidence_from_method(method: str) -> float:
    if method == "exact_norm":
        return 0.97
    if method == "regex_alias":
        return 0.90
    return 0.70


def _resolve_matches_for_posts(
    posts: pd.DataFrame, group_alias_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Internal helper that creates post-level resolution objects.
    """
    alias_match_rows: list[dict[str, Any]] = []

    for _, alias_row in group_alias_df.iterrows():
        alias = str(alias_row["alias"])
        alias_norm = str(alias_row["alias_norm"])
        pattern = alias_row["pattern"]

        exact_norm_mask = posts["text_norm"].eq(alias_norm)
        regex_mask = posts["text"].str.contains(pattern, na=False)

        matched_idx = posts.index[regex_mask | exact_norm_mask].tolist()

        for idx in matched_idx:
            alias_match_rows.append(
                {
                    "post_idx": idx,
                    "post_id": posts.at[idx, "post_id"],
                    "group_name_norm": alias_row["group_name_norm"],
                    "group_name": alias_row["group_name"],
                    "entity_type": alias_row["entity_type"],
                    "alias": alias,
                    "alias_norm": alias_norm,
                    "candidate_entity_ids": list(alias_row["candidate_entity_ids"]),
                    "canonical_names": list(alias_row["canonical_names"]),
                    "match_method": (
                        "exact_norm"
                        if bool(exact_norm_mask.get(idx, False))
                        else "regex_alias"
                    ),
                    "mention_len": len(alias),
                }
            )

    if not alias_match_rows:
        return pd.DataFrame(
            columns=["post_id", "resolved_entities", "n_resolved_groups"]
        )

    alias_matches = pd.DataFrame(alias_match_rows)

    # Keep strongest match per post per group.
    alias_matches_ranked = alias_matches.sort_values(
        ["post_id", "group_name_norm", "mention_len", "match_method"],
        ascending=[True, True, False, True],
    ).drop_duplicates(subset=["post_id", "group_name_norm"], keep="first")

    resolution_rows: list[dict[str, Any]] = []

    for post_id, sub in alias_matches_ranked.groupby("post_id"):
        resolved_entities = []

        for _, row in sub.iterrows():
            resolved_entities.append(
                {
                    "group_name": row["group_name"],
                    "group_name_norm": row["group_name_norm"],
                    "mention_text": row["alias"],
                    "confidence": _confidence_from_method(row["match_method"]),
                    "resolution_method": row["match_method"],
                    "candidate_entity_ids": row["candidate_entity_ids"],
                    "candidate_canonical_names": row["canonical_names"],
                }
            )

        resolution_rows.append(
            {
                "post_id": post_id,
                "resolved_entities": resolved_entities,
                "n_resolved_groups": len(resolved_entities),
            }
        )

    return pd.DataFrame(resolution_rows)


def resolve_entities(posts: pd.DataFrame, group_alias_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve grouped entities in post text using rules-first matching.

    Output:
        posts_resolved dataframe with:
            text_norm
            resolved_entities
            n_resolved_groups
    """
    posts_resolved = posts.copy()
    posts_resolved["text"] = posts_resolved["text"].fillna("").astype(str)
    posts_resolved["text_norm"] = posts_resolved["text"].apply(normalize_text)

    post_resolutions = _resolve_matches_for_posts(posts_resolved, group_alias_df)

    posts_resolved = posts_resolved.merge(post_resolutions, on="post_id", how="left")
    posts_resolved["resolved_entities"] = posts_resolved["resolved_entities"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    posts_resolved["n_resolved_groups"] = (
        posts_resolved["n_resolved_groups"].fillna(0).astype(int)
    )

    return posts_resolved


def has_group(resolved_entities: list[dict[str, Any]], group_name: str) -> bool:
    """
    Check whether a resolved entity list contains a target grouped entity.
    """
    return any(entity.get("group_name") == group_name for entity in resolved_entities)


def filter_posts_for_entity(
    posts_resolved: pd.DataFrame, entity_name: str
) -> pd.DataFrame:
    """
    Return only posts that match a given grouped entity.
    """
    mask = posts_resolved["resolved_entities"].apply(
        lambda ents: has_group(ents, entity_name)
    )
    return posts_resolved.loc[mask].copy().reset_index(drop=True)


def build_entity_match_summary(posts_resolved: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple grouped-entity match summary from resolved posts.

    Output columns:
        entity_name
        matched_posts
    """
    rows: list[dict[str, Any]] = []

    all_group_names = sorted(
        {
            entity["group_name"]
            for resolved in posts_resolved["resolved_entities"]
            for entity in resolved
        }
    )

    for group_name in all_group_names:
        matched_posts = int(
            posts_resolved["resolved_entities"]
            .apply(lambda ents: has_group(ents, group_name))
            .sum()
        )
        rows.append({"entity_name": group_name, "matched_posts": matched_posts})

    summary_df = (
        pd.DataFrame(rows)
        .sort_values("matched_posts", ascending=False)
        .reset_index(drop=True)
    )
    return summary_df
