from src.config import DATA_DIR
from src.data_loader import load_all
from src.entity_resolution import (
    build_group_alias_table,
    build_grouped_entities,
    filter_posts_for_entity,
    resolve_entities,
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

posts, authors, entities = load_all(DATA_DIR)

grouped_entities = build_grouped_entities(entities)
group_alias_df = build_group_alias_table(grouped_entities)
posts_resolved = resolve_entities(posts, group_alias_df)

pfizer_posts = filter_posts_for_entity(posts_resolved, "Pfizer")
pfizer_posts = add_text_clean(pfizer_posts, clean_text_for_clustering)
pfizer_posts = assign_primary_theme(pfizer_posts)
pfizer_posts = flag_weak_signal_posts(pfizer_posts)

qualified = filter_theme_qualified_posts(pfizer_posts)
clustered = cluster_within_themes(qualified)

labels = build_narrative_labels(clustered)
labels = mark_probably_junk_narratives(labels)
usable = filter_usable_narratives(labels)

usable_posts = attach_only_usable_posts(clustered, usable)

final_narratives, narrative_posts_scored = build_final_narratives(
    usable_posts,
    usable,
    authors,
)

print("Final narratives:")
print(
    final_narratives[
        [
            "narrative_id",
            "primary_theme",
            "label",
            "n_posts",
            "n_unique_authors",
            "risk_score",
            "confidence_score",
            "driver_tags",
        ]
    ].head(15)
)

print("\nScored narrative posts:")
print(
    narrative_posts_scored[
        [
            "post_id",
            "narrative_id",
            "weighted_engagement",
            "risk_term_hits",
            "risk_terms_matched",
        ]
    ].head(10)
)
