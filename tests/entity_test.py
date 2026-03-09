from src.config import DATA_DIR
from src.data_loader import load_all
from src.entity_resolution import (
    build_group_alias_table,
    build_grouped_entities,
    build_entity_match_summary,
    filter_posts_for_entity,
    resolve_entities,
)

posts, authors, entities = load_all(DATA_DIR)

grouped_entities = build_grouped_entities(entities)
group_alias_df = build_group_alias_table(grouped_entities)
posts_resolved = resolve_entities(posts, group_alias_df)

print("Grouped entities:")
print(grouped_entities[["entity_id", "canonical_name", "group_name"]])

print("\nAlias table:")
print(group_alias_df[["group_name", "alias"]].head(20))

print("\nResolved posts shape:", posts_resolved.shape)
print("Posts with matches:", (posts_resolved["n_resolved_groups"] > 0).sum())

summary = build_entity_match_summary(posts_resolved)
print("\nEntity match summary:")
print(summary.head(10))

pfizer_posts = filter_posts_for_entity(posts_resolved, "Pfizer")
print("\nPfizer matched posts:", len(pfizer_posts))
print(pfizer_posts[["post_id", "platform", "created_at", "text"]].head(5))
