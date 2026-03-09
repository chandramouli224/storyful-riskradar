# Key Design Decisions

This document summarizes the main design decisions, trade-offs, and areas for improvement in the Narrative Risk Explorer prototype.

The system was designed as an analyst triage tool, not a fully automated narrative detection system. The primary goal is to help analysts quickly identify potentially risky narratives while maintaining transparency and interpretability.

## 1. Rules-Based Entity Resolution

### Decision

Entity matching is performed using a rules-first alias grouping approach rather than embedding-based matching.

**Example:**

- Pfizer
- Pfizer Inc
- pfizer

All resolve to the canonical entity Pfizer.

### Why

Embedding-based entity matching can introduce false positives, especially when entity names overlap with common terms or unrelated contexts.

A deterministic alias grouping approach provides:

- higher precision
- predictable behavior
- easier debugging

This is especially important in analyst-facing systems where incorrect entity matching can undermine trust.

### Trade-Off

The rules-based approach requires maintaining alias mappings manually.

In large production systems this could be augmented with:

- embedding similarity fallback
- knowledge graph entity linking
- external entity resolution services

## 2. Theme-First Narrative Clustering

### Decision

Posts are first classified into high-level themes before clustering.

**Themes include:**

- product_clinical
- safety_quality
- regulatory_legal
- financial_market
- political_reputational
- executive_corporate

Clustering is performed within each theme rather than across the full dataset.

### Why

Clustering the entire dataset often groups unrelated discussions together.

For example:

- a financial discussion
- a regulatory lawsuit discussion

may reference the same company but represent different narratives.

Theme-first segmentation improves cluster coherence and makes narratives easier to interpret.

### Trade-Off

Theme classification errors can propagate into clustering results.

Future improvements could include:

- multi-label theme assignment
- transformer-based topic classification
- dynamic topic discovery

## 3. Narrative Scoring Model

### Decision

Narratives are ranked using a composite scoring function combining multiple signals.

**Signals include:**

- Volume
- Velocity
- Engagement
- Author Influence
- Risk Language
- Theme Severity

### Why

A single metric (such as engagement) is insufficient for narrative risk detection.

The composite score provides a more balanced view of narrative impact and potential risk.

Additionally, breaking the score into individual components allows analysts to understand why a narrative was prioritized.

### Trade-Off

The current scoring weights are heuristic and not calibrated using labeled data.

In production systems these weights could be learned using:

- historical analyst decisions
- supervised learning models
- reinforcement learning from feedback

## 4. Evidence Selection Strategy

### Decision

Each narrative surfaces 2–3 representative posts as evidence.

Evidence posts are selected using a combination of:

- semantic similarity to the narrative centroid
- engagement metrics
- risk language signals

### Why

Presenting entire clusters to analysts would be inefficient.

Evidence posts provide a quick snapshot of the narrative context.

### Trade-Off

Evidence selection may occasionally miss important outlier posts.

Future improvements could include:

- diversity-aware evidence selection
- summarization models
- narrative-level summaries

## 5. Human-in-the-Loop Feedback

### Decision

The Streamlit interface allows analysts to record feedback on:

- incorrect entity matches
- risk score calibration

Feedback is stored locally as JSON records.

### Why

Narrative risk detection is inherently subjective and context-dependent.

Human feedback allows the system to evolve over time.

### Trade-Off

Feedback is currently stored but not used for automated retraining.

Future work could integrate:

- active learning
- weak supervision frameworks
- reinforcement learning approaches

## If I Had More Time

Several improvements could significantly strengthen the system.

### 1. Narrative Evolution Tracking

Track narratives across time windows to detect:

- narrative growth
- narrative merging
- narrative decline

### 2. Improved Clustering

Replace current clustering with density-based approaches such as:

HDBSCAN

This would allow the system to detect narratives of varying sizes without requiring fixed cluster structures.

### 3. Better Risk Language Detection

The current risk language detection uses simple keyword matching.

Future improvements could include:

- transformer-based toxicity models
- legal / regulatory risk classifiers
- contextual sentiment models

### 4. Cross-Platform Narrative Linking

Currently narratives are platform-agnostic but not explicitly linked across platforms.

Future work could detect when narratives propagate across:

- Twitter/X
- Facebook
- Reddit
- news sources

### 5. Evaluation Framework

A larger labeled dataset would enable:

- precision/recall evaluation
- ranking performance measurement
- narrative detection accuracy analysis

## Summary

This prototype demonstrates a modular architecture for narrative monitoring that prioritizes:

- interpretability
- analyst usability
- evidence-based analysis

The system is intentionally designed as a decision-support tool rather than an automated moderation system.
