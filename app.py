import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Narrative Risk Explorer",
    layout="wide",
)

# -----------------------------
# PATHS
# -----------------------------
OUTPUT_DIR = Path("outputs")
FEEDBACK_PATH = Path("feedback/feedback.jsonl")


# -----------------------------
# HELPERS
# -----------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def parse_driver_tags(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def make_narrative_display_label(row):
    label = str(row.get("label", "Untitled narrative"))
    theme = str(row.get("theme", "unknown"))
    n_posts = int(row.get("n_posts", 0))
    risk = safe_float(row.get("risk_score", 0.0))
    conf = safe_float(row.get("confidence_score", 0.0))
    return f"{label} | {theme} | risk {risk:.1f} | conf {conf:.1f} | {n_posts} posts"


def narrative_summary_text(row):
    theme = str(row.get("theme", "unknown")).replace("_", " ")
    label = str(row.get("label", "Untitled narrative"))
    n_posts = int(row.get("n_posts", 0))
    n_authors = int(row.get("n_unique_authors", 0))
    risk = safe_float(row.get("risk_score", 0.0))
    conf = safe_float(row.get("confidence_score", 0.0))

    return (
        f"This narrative is categorized as **{theme}** and labeled **{label}**. "
        f"It contains **{n_posts} posts** from **{n_authors} unique authors**. "
        f"The current **risk score is {risk:.1f}** and **confidence is {conf:.1f}**."
    )


def ensure_feedback_file():
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_PATH.exists():
        FEEDBACK_PATH.touch()


@st.cache_data
def load_data():
    entity_df = pd.read_csv(OUTPUT_DIR / "entity_overview.csv")
    narratives_df = pd.read_csv(OUTPUT_DIR / "narratives.csv")
    evidence_df = pd.read_csv(OUTPUT_DIR / "narrative_evidence.csv")
    posts_df = pd.read_csv(OUTPUT_DIR / "narrative_posts.csv")
    return entity_df, narratives_df, evidence_df, posts_df


# -----------------------------
# LOAD DATA
# -----------------------------
entity_df, narratives_df, evidence_df, posts_df = load_data()

# -----------------------------
# APP HEADER
# -----------------------------
st.title("Narrative Risk Explorer")
st.caption(
    "This is an analyst triage system, not an autonomous decision-maker. "
    "It prioritizes narratives for review, shows why they are risky, and preserves uncertainty and evidence."
)

# -----------------------------
# SIDEBAR ENTITY SELECTION
# -----------------------------
st.sidebar.header("Entity Selection")

entities = sorted(entity_df["entity_name"].dropna().unique().tolist())

selected_entity = st.sidebar.selectbox(
    "Choose entity",
    entities,
)

entity_info = entity_df[entity_df["entity_name"] == selected_entity].iloc[0]

st.sidebar.markdown("### Match Overview")
st.sidebar.metric("Matched Posts", int(entity_info["matched_posts"]))
st.sidebar.metric(
    "% Posts Matched", f"{safe_float(entity_info['pct_posts_matched']):.2f}"
)
st.sidebar.metric("Narratives Detected", int(entity_info["n_narratives"]))
st.sidebar.metric("Avg Confidence", f"{safe_float(entity_info['avg_confidence']):.1f}")

if "notes" in entity_info.index and pd.notna(entity_info["notes"]):
    st.sidebar.caption(str(entity_info["notes"]))

# -----------------------------
# FILTER ENTITY DATA
# -----------------------------
entity_narratives = narratives_df[
    narratives_df["entity_name"] == selected_entity
].copy()

entity_narratives = entity_narratives.sort_values(
    ["risk_score", "confidence_score"],
    ascending=[False, False],
).reset_index(drop=True)

entity_evidence = evidence_df[evidence_df["entity_name"] == selected_entity].copy()

entity_posts = posts_df[posts_df["entity_name"] == selected_entity].copy()

# -----------------------------
# NARRATIVE LIST
# -----------------------------
st.header("Narratives ranked by risk")

table_df = entity_narratives.copy()
table_df["driver_tags_display"] = table_df["driver_tags"].apply(
    lambda x: ", ".join(parse_driver_tags(x))
)

display_cols = [
    "theme",
    "label",
    "n_posts",
    "n_unique_authors",
    "risk_score",
    "confidence_score",
    "driver_tags_display",
]

st.dataframe(
    table_df[display_cols],
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# NARRATIVE SELECTOR
# -----------------------------
st.subheader("Select Narrative")

entity_narratives["narrative_display"] = entity_narratives.apply(
    make_narrative_display_label,
    axis=1,
)

selected_display = st.selectbox(
    "Narrative",
    entity_narratives["narrative_display"].tolist(),
)

narrative_row = entity_narratives[
    entity_narratives["narrative_display"] == selected_display
].iloc[0]

selected_narrative = narrative_row["narrative_id"]

# -----------------------------
# NARRATIVE DETAIL
# -----------------------------
st.header("Narrative Detail")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk Score", f"{safe_float(narrative_row['risk_score']):.1f}")
c2.metric("Confidence", f"{safe_float(narrative_row['confidence_score']):.1f}")
c3.metric("Posts", int(narrative_row["n_posts"]))
c4.metric("Authors", int(narrative_row.get("n_unique_authors", 0)))

st.markdown("### Summary")
st.write(narrative_summary_text(narrative_row))

left, right = st.columns([1, 2])

with left:
    st.markdown("### Theme")
    st.write(str(narrative_row["theme"]).replace("_", " "))

    st.markdown("### Label")
    st.write(narrative_row["label"])

with right:
    st.markdown("### Driver Tags")
    driver_tags = parse_driver_tags(narrative_row["driver_tags"])
    st.write(driver_tags if driver_tags else ["No driver tags available"])

# -----------------------------
# DRIVER BREAKDOWN
# -----------------------------
st.markdown("### Risk Driver Components")

driver_cols = [
    "volume_norm",
    "velocity_norm",
    "engagement_norm",
    "author_norm",
    "language_risk_norm",
    "theme_prior_norm",
]

driver_labels = {
    "volume_norm": "Volume",
    "velocity_norm": "Velocity",
    "engagement_norm": "Engagement",
    "author_norm": "Author Influence",
    "language_risk_norm": "Risk Language",
    "theme_prior_norm": "Theme Prior",
}

driver_df = pd.DataFrame(
    {
        "driver": [driver_labels[c] for c in driver_cols],
        "value": [safe_float(narrative_row[c]) for c in driver_cols],
    }
)

st.bar_chart(driver_df.set_index("driver"))

# -----------------------------
# NARRATIVE MEMBER POSTS
# -----------------------------
with st.expander("Show narrative member posts"):
    member_posts = entity_posts[
        entity_posts["narrative_id"] == selected_narrative
    ].copy()

    show_cols = [
        c for c in ["created_at", "platform", "text"] if c in member_posts.columns
    ]
    st.dataframe(
        member_posts[show_cols].head(20),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# EVIDENCE POSTS
# -----------------------------
st.header("Evidence Posts")

evidence = entity_evidence[entity_evidence["narrative_id"] == selected_narrative].copy()

if "post_id" in evidence.columns:
    evidence = evidence.drop_duplicates(subset=["post_id"], keep="first")

evidence = evidence.drop_duplicates(subset=["text"], keep="first")
evidence = evidence.sort_values("evidence_rank").head(3)

if evidence.empty:
    st.info("No evidence posts available for this narrative.")
else:
    for _, row in evidence.iterrows():
        st.markdown("---")
        st.write(row["text"])
        st.caption(
            f"Platform: {row['platform']} | "
            f"Engagement: {safe_float(row.get('weighted_engagement', 0)):.3f} | "
            f"Risk terms: {int(row.get('risk_term_hits', 0))}"
        )

# -----------------------------
# FEEDBACK
# -----------------------------
st.header("Feedback")

ensure_feedback_file()

st.markdown("### 1) Entity Match Feedback")

entity_feedback_type = st.selectbox(
    "Entity match assessment",
    [
        "Looks correct",
        "Incorrect entity match",
    ],
    key="entity_feedback_type",
)

correct_entity = None
if entity_feedback_type == "Incorrect entity match":
    entity_options = ["none"] + entities
    correct_entity = st.selectbox(
        "Select correct entity",
        entity_options,
        key="correct_entity",
    )

st.markdown("### 2) Risk Score Feedback")

risk_feedback = st.radio(
    "Is the risk score appropriate?",
    ["Too high", "About right", "Too low"],
    horizontal=True,
    key="risk_feedback",
)

note = st.text_area("Optional note")

if st.button("Submit Feedback"):
    feedback_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "entity": selected_entity,
        "narrative_id": selected_narrative,
        "label": str(narrative_row["label"]),
        "theme": str(narrative_row["theme"]),
        "risk_score": safe_float(narrative_row["risk_score"]),
        "confidence_score": safe_float(narrative_row["confidence_score"]),
        "entity_feedback_type": entity_feedback_type,
        "correct_entity": (
            correct_entity
            if entity_feedback_type == "Incorrect entity match"
            else selected_entity
        ),
        "risk_feedback": risk_feedback,
        "note": note,
    }

    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_record) + "\n")

    st.success("Feedback saved.")
