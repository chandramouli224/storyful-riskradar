from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR, OUTPUT_DIR
from src.data_loader import load_all
from src.entity_resolution import (
    build_entity_match_summary,
    build_group_alias_table,
    build_grouped_entities,
    filter_posts_for_entity,
    resolve_entities,
)
from src.evidence import deduplicate_evidence, select_evidence_posts
from src.export_artifacts import (
    build_entity_overview,
    prepare_evidence_export,
    prepare_narratives_export,
    prepare_posts_export,
    write_artifacts,
)
from src.narrative_builder import (
    attach_only_usable_posts,
    build_narrative_labels,
    cluster_within_themes,
    filter_usable_narratives,
    flag_weak_signal_posts,
    mark_probably_junk_narratives,
)
from src.scoring import build_final_narratives
from src.theme_tagging import (
    add_text_clean,
    assign_primary_theme,
    filter_theme_qualified_posts,
)
from src.utils import clean_text_for_clustering


def run_for_entity(
    entity_name: str,
    posts_resolved: pd.DataFrame,
    authors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full pipeline for one selected entity.

    Returns:
        entity_overview_df
        narratives_export
        evidence_export
        posts_export
    """
    entity_posts = filter_posts_for_entity(posts_resolved, entity_name)

    if entity_posts.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    entity_posts = add_text_clean(entity_posts, clean_text_for_clustering)
    entity_posts = assign_primary_theme(entity_posts)
    entity_posts = flag_weak_signal_posts(entity_posts)

    qualified_posts = filter_theme_qualified_posts(entity_posts)
    if qualified_posts.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    clustered_posts = cluster_within_themes(qualified_posts)
    if clustered_posts.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    narrative_labels = build_narrative_labels(clustered_posts)
    narrative_labels = mark_probably_junk_narratives(narrative_labels)
    usable_narratives = filter_usable_narratives(narrative_labels)

    if usable_narratives.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    usable_posts = attach_only_usable_posts(clustered_posts, usable_narratives)

    final_narratives, narrative_posts_scored = build_final_narratives(
        usable_posts,
        usable_narratives,
        authors,
    )

    evidence_df = select_evidence_posts(narrative_posts_scored)
    evidence_df = deduplicate_evidence(evidence_df)

    entity_overview_df = build_entity_overview(
        posts_resolved, final_narratives, entity_name
    )
    narratives_export = prepare_narratives_export(final_narratives, entity_name)
    evidence_export = prepare_evidence_export(evidence_df, entity_name)
    posts_export = prepare_posts_export(narrative_posts_scored, entity_name)

    return entity_overview_df, narratives_export, evidence_export, posts_export


def main() -> None:
    posts, authors, entities = load_all(DATA_DIR)

    grouped_entities = build_grouped_entities(entities)
    group_alias_df = build_group_alias_table(grouped_entities)
    posts_resolved = resolve_entities(posts, group_alias_df)

    entity_summary = build_entity_match_summary(posts_resolved)

    # Keep entities with enough matched posts to produce usable narratives.
    selected_entities = entity_summary.loc[
        entity_summary["matched_posts"] >= 10, "entity_name"
    ].tolist()

    all_entity_overview = []
    all_narratives = []
    all_evidence = []
    all_posts = []

    for entity_name in selected_entities:
        print(f"Running pipeline for entity: {entity_name}")

        entity_overview_df, narratives_export, evidence_export, posts_export = (
            run_for_entity(
                entity_name=entity_name,
                posts_resolved=posts_resolved,
                authors=authors,
            )
        )

        if not entity_overview_df.empty:
            all_entity_overview.append(entity_overview_df)
        if not narratives_export.empty:
            all_narratives.append(narratives_export)
        if not evidence_export.empty:
            all_evidence.append(evidence_export)
        if not posts_export.empty:
            all_posts.append(posts_export)

    entity_overview_out = (
        pd.concat(all_entity_overview, ignore_index=True)
        if all_entity_overview
        else pd.DataFrame()
    )
    narratives_out = (
        pd.concat(all_narratives, ignore_index=True)
        if all_narratives
        else pd.DataFrame()
    )
    evidence_out = (
        pd.concat(all_evidence, ignore_index=True) if all_evidence else pd.DataFrame()
    )
    posts_out = pd.concat(all_posts, ignore_index=True) if all_posts else pd.DataFrame()

    write_artifacts(
        output_dir=OUTPUT_DIR,
        entity_overview_df=entity_overview_out,
        narratives_df=narratives_out,
        evidence_df=evidence_out,
        posts_df=posts_out,
    )

    print("\nPipeline complete.")
    print(f"Entities exported: {entity_overview_out.shape[0]}")
    print(f"Narratives exported: {narratives_out.shape[0]}")
    print(f"Evidence rows exported: {evidence_out.shape[0]}")
    print(f"Narrative posts exported: {posts_out.shape[0]}")


if __name__ == "__main__":
    main()
