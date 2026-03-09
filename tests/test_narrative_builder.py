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
print("Qualified posts:", len(qualified))

clustered = cluster_within_themes(qualified)
print("Clustered posts:", len(clustered))
print(
    clustered[["post_id", "primary_theme", "theme_cluster_id", "narrative_id"]].head(10)
)

labels = build_narrative_labels(clustered)
labels = mark_probably_junk_narratives(labels)

print("\nNarrative labels:")
print(labels.head(20))

usable = filter_usable_narratives(labels)
print("\nUsable narratives:")
print(usable)

usable_posts = attach_only_usable_posts(clustered, usable)
print("\nUsable posts:", len(usable_posts))
print(usable_posts[["post_id", "primary_theme", "narrative_id", "text"]].head(10))
