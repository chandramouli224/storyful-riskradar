from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEEDBACK_PATH = PROJECT_ROOT / "feedback" / "feedback.jsonl"

# -----------------------------
# ENTITY RESOLUTION
# -----------------------------
CORP_SUFFIXES = [
    "inc",
    "plc",
    "co",
    "corp",
    "corporation",
    "ltd",
    "llc",
    "company",
    "group",
]

# Keep these separate even if suffix stripping would collapse them.
MANUAL_GROUP_OVERRIDES = {
    "merck": "Merck",
    "merck_co_inc": "Merck & Co., Inc.",
}

# -----------------------------
# THEME TAGGING
# -----------------------------
THEME_LEXICON = {
    "regulatory_legal": [
        "fda",
        "ema",
        "mhra",
        "investigation",
        "lawsuit",
        "legal",
        "court",
        "fine",
        "settlement",
        "regulator",
        "approved",
        "approval",
        "ban",
        "recall",
        "compliance",
        "charged",
        "hearing",
        "senate",
        "congress",
        "subpoena",
        "mandate",
        "policy",
        "ruling",
        "petition",
        "liability",
        "violation",
    ],
    "safety_quality": [
        "side effect",
        "side effects",
        "adverse",
        "adverse event",
        "injury",
        "death",
        "deaths",
        "harm",
        "unsafe",
        "safety",
        "risk",
        "contamination",
        "quality issue",
        "defect",
        "warning",
        "hospitalized",
        "myocarditis",
        "blood clot",
        "stroke",
        "seizure",
        "vaccine injury",
        "fatal",
        "toxicity",
        "reaction",
    ],
    "financial_market": [
        "stock",
        "stocks",
        "shares",
        "earnings",
        "revenue",
        "profit",
        "sales",
        "market cap",
        "guidance",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
        "investor",
        "analyst",
        "portfolio",
        "deal",
        "acquisition",
        "bid",
        "valuation",
        "forecast",
        "price target",
        "bullish",
        "bearish",
        "call option",
        "puts",
        "market",
    ],
    "executive_corporate": [
        "ceo",
        "chairman",
        "board",
        "executive",
        "management",
        "company",
        "firm",
        "leadership",
        "hiring",
        "jobs",
        "investment",
        "expansion",
        "facility",
        "manufacturing",
        "plant",
        "office",
        "role",
        "director",
        "chief",
        "head of",
        "employee",
        "workshop",
        "partnership",
        "associate",
    ],
    "product_clinical": [
        "drug",
        "vaccine",
        "vaccines",
        "booster",
        "boosters",
        "trial",
        "trials",
        "phase 1",
        "phase 2",
        "phase 3",
        "study",
        "studies",
        "clinical",
        "patients",
        "treatment",
        "therapy",
        "medicine",
        "oncology",
        "obesity",
        "indication",
        "data",
        "efficacy",
        "dose",
        "doses",
        "mrna",
        "immunotherapy",
        "disease",
    ],
    "political_reputational": [
        "trump",
        "biden",
        "fauci",
        "cdc",
        "who",
        "hoax",
        "corruption",
        "crime",
        "scandal",
        "media",
        "cover up",
        "cover-up",
        "propaganda",
        "lobbyist",
        "greene",
        "kennedy",
        "rfk",
        "deep state",
        "big pharma",
        "government",
        "politics",
        "political",
        "senator",
        "democrat",
        "republican",
    ],
}

THEME_PRIORITY = {
    "safety_quality": 6,
    "regulatory_legal": 5,
    "political_reputational": 4,
    "financial_market": 3,
    "product_clinical": 2,
    "executive_corporate": 1,
    "misc": 0,
}

THEME_PRIOR = {
    "safety_quality": 1.00,
    "regulatory_legal": 0.95,
    "political_reputational": 0.85,
    "product_clinical": 0.70,
    "executive_corporate": 0.50,
    "financial_market": 0.45,
}

# -----------------------------
# RISK LEXICON
# -----------------------------
RISK_LEXICON = {
    "regulatory_legal": [
        "lawsuit",
        "legal",
        "court",
        "fine",
        "charged",
        "hearing",
        "investigation",
        "recall",
        "subpoena",
        "violation",
        "settlement",
        "regulator",
        "fda",
        "ema",
    ],
    "safety_quality": [
        "side effect",
        "side effects",
        "adverse",
        "adverse event",
        "injury",
        "death",
        "deaths",
        "unsafe",
        "harm",
        "risk",
        "myocarditis",
        "blood clot",
        "stroke",
        "vaccine injury",
        "toxicity",
        "reaction",
        "fatal",
    ],
    "reputation_political": [
        "hoax",
        "corruption",
        "crime",
        "scandal",
        "cover up",
        "cover-up",
        "deep state",
        "big pharma",
        "propaganda",
        "fraud",
        "lied",
        "shame",
        "boycott",
    ],
}

# -----------------------------
# NARRATIVE CLEANING
# -----------------------------
JUNK_LABEL_TERMS = {
    "vs",
    "fc",
    "2025",
    "schedule",
    "weather",
    "latest",
    "today",
    "new",
    "did",
    "great",
    "little",
    "watch",
    "make",
    "going",
    "time",
    "people",
    "years",
    "world",
    "business",
    "special",
    "said",
    "drop",
    "bad",
    "love",
    "public",
}

JUNK_TEXT_PATTERNS = [
    r"\bweather\b",
    r"\bschedule\b",
    r"\bfootball\b",
    r"\bfc\b",
    r"\brankings\b",
    r"\bzverev\b",
    r"\bzuckerberg\b",
    r"\bzeta\b",
    r"\bzebby\b",
]

# -----------------------------
# CLUSTERING
# -----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_DISTANCE_THRESHOLD = 0.60
MIN_USABLE_NARRATIVE_SIZE = 2

# -----------------------------
# SCORING
# -----------------------------
SCORING_WEIGHTS = {
    "volume": 0.20,
    "velocity": 0.20,
    "engagement": 0.20,
    "author": 0.10,
    "language_risk": 0.20,
    "theme_prior": 0.10,
}

CONFIDENCE_WEIGHTS = {
    "coherence": 0.30,
    "size": 0.25,
    "follower_cov": 0.15,
    "handle_cov": 0.15,
    "author_diversity": 0.15,
}
