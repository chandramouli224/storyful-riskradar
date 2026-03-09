from src.config import DATA_DIR
from src.data_loader import load_all
from src.entity_resolution import (
    build_group_alias_table,
    build_grouped_entities,
    filter_posts_for_entity,
    resolve_entities,
)
from src.theme_tagging import (
    add_text_clean,
    assign_primary_theme,
    build_theme_distribution,
    filter_theme_qualified_posts,
)
from src.utils import clean_text_for_clustering

posts, authors, entities = load_all(DATA_DIR)

grouped_entities = build_grouped_entities(entities)
group_alias_df = build_group_alias_table(grouped_entities)
posts_resolved = resolve_entities(posts, group_alias_df)

pfizer_posts = filter_posts_for_entity(posts_resolved, "Pfizer")
pfizer_posts = add_text_clean(pfizer_posts, clean_text_for_clustering)
pfizer_themed = assign_primary_theme(pfizer_posts)

print("Pfizer posts:", len(pfizer_posts))
print("\nTheme distribution:")
print(build_theme_distribution(pfizer_themed))

qualified = filter_theme_qualified_posts(pfizer_themed)
print("\nTheme-qualified posts:", len(qualified))

print("\nSample themed posts:")
print(
    pfizer_themed[
        ["post_id", "primary_theme", "primary_theme_score", "matched_themes", "text"]
    ].head(10)
)
