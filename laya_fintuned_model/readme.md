# Fine-Tuning Laya for App/Yelp Sentiment Classification

## Table of Contents

1. [Overview](#1-overview)
2. [What is a System One Model?](#2-what-is-a-system-one-model)
3. [About the Laya Model](#3-about-the-laya-model)
4. [The Dataset: Mixed App Store & Yelp Reviews](#4-the-dataset-mixed-app-store--yelp-reviews)
5. [Converting the Dataset to Laya JSONL Format](#5-converting-the-dataset-to-laya-jsonl-format)
6. [Fine-Tuning Strategy and Training Details](#6-fine-tuning-strategy-and-training-details)
7. [Evaluation Results](#7-evaluation-results)
8. [Why Laya is Fast](#8-why-laya-is-fast)
9. [Lessons Learned and Caveats](#9-lessons-learned-and-caveats)
10. [Project Files and Artifacts](#10-project-files-and-artifacts)
11. [Reproducing the Pipeline](#11-reproducing-the-pipeline)
12. [Summary](#12-summary)

---

## 1. Overview

This document describes the complete process of fine-tuning **Laya**, an open-source
"System One" decision model, on a binary sentiment classification task using a mixed
App Store and Yelp review dataset. The goal was to adapt a general-purpose decision
engine into a domain-specific sentiment classifier for short, informal user-generated
reviews.

**Final Results (Held-Out Test Set, 5,000 samples):**

| Metric | Score |
|---|---|
| Accuracy | 0.8880 |
| Precision (macro) | 0.8867 |
| Recall (macro) | 0.8890 |
| F1-Score (macro) | 0.8874 |
| Negative F1 | 0.8914 |
| Positive F1 | 0.8755 |

The fine-tuned model achieved **~88.7% macro F1**, a substantial improvement over the
base checkpoint's zero-shot performance on this task, and competitive with fine-tuned
BERT-family baselines on similar Yelp sentiment datasets.

**Key numbers at a glance:**

| Item | Value |
|---|---|
| Base model | Laya multilingual (mmBERT-base encoder) |
| Base model size | ~322M parameters [citation:2][citation:14] |
| Trainable params (test profile) | ~45M (top encoder layers + decision head) [citation:1] |
| Training rows (after internal split) | 155,348 |
| Held-out test rows | 5,000 (sampled from 43,152) |
| Training time | ~1 h 30 min (1 epoch, T4, batch 8) |
| Final accuracy | 0.8880 |
| Final macro F1 | 0.8874 |
| Inference latency | ~30–50 ms/review |
| Model size on disk | ~1.2 GB |

---

## 2. What is a System One Model?

Traditional large language models (LLMs) are **generative**: they produce answers
token by token, which is slow and introduces the risk of hallucination or unparseable
output. A **System One decision model** takes a fundamentally different approach:

- **No text generation.** It does not write an explanation or a paragraph. It directly
  outputs a **typed decision** [citation:19].
- **Single forward pass.** Given a state (text) and a question, it evaluates all
  candidate answers in parallel and returns a probability distribution over them in
  one pass.
- **Typed outputs.** It supports three decision primitives [citation:2]:
  - **`choice`**: pick one label from a list you define (e.g., "Positive" vs. "Negative").
  - **`score`**: return an expected level on an ordinal rubric you write (e.g., urgency
    from "not urgent" through "critical").
  - **`noul`** (yes/no): return a calibrated probability that a statement is true.

Because there is no autoregressive decoding, System One models are **extremely fast**.
The key property is that the model can only answer within the options you define, so it
cannot invent a label that does not exist. There is no JSON to parse and no retry loop
when parsing fails [citation:19].

The term "System One" is borrowed from Daniel Kahneman's model of fast, intuitive
thinking. It describes a model that makes quick, bounded judgments rather than
open-ended generation.

---

## 3. About the Laya Model

### 3.1 What Laya Is

**Laya** is an open-source, Apache-2.0 licensed, non-autoregressive System One decision
engine developed by Convai Innovations [citation:8][citation:19]. It evaluates typed
questions over any state (text, email, ticket, or JSON document) in a single forward
pass. It is not a generative LLM and does not write text. Instead, it returns
**calibrated probabilities over predefined answer options**.

Laya was released as an open-weights alternative to TypeSafe's hosted Jev API. Both
belong to the "System One" category: models designed for classification, routing, and
bounded decision tasks rather than open-ended generation [citation:19].

### 3.2 The Three Checkpoints

Laya ships with three checkpoints, and a `Router` that picks between them per request
based on detected script and language [citation:2][citation:8]:

| Checkpoint | Encoder | Params | Context | Use it for |
|---|---|---|---|---|
| `laya` (English) | ModernBERT-large | 421M | 512 | English text |
| `laya-multilingual` | mmBERT-base | 322M | 1,024 (up to 8,192 with `max_len`) | 100+ languages, 2× faster |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1,024 | Typed-decisions workflows |

In this project, the **multilingual** checkpoint was used. This was not a deliberate
choice: the `laya-finetune` training script automatically selected the `multilingual`
subfolder when given the base repository ID `convaiinnovations/laya` [citation:1].

### 3.3 Internal Architecture: The DecisionModel

The encoder is paired with a **decision head** that scores each candidate answer. For a
`choice` question with two options ("Negative" and "Positive"), the model constructs a
single sequence containing:

1. The input text (`state`).
2. The question instructions.
3. Special marker tokens for each candidate option.

The encoder processes this entire sequence in **one forward pass**. The marker positions
(`marker_pos`), a marker mask (`marker_mask`), and a question-type indicator (`qtype`)
are fed into the `DecisionModel.forward()` method, which produces logits for each
option. The softmax of these logits gives the probability distribution over labels
[citation:13].

This is why a manual training loop that calls `model(input_ids, attention_mask)`
directly will fail: it omits the marker positions and question type, so the model
cannot know **which tokens correspond to which answer**.

### 3.4 How Laya Differs from Generative LLMs

| Property | Generative LLM | Laya (System One) |
|---|---|---|
| Output | Free text, needs parsing | Typed answer (choice/score/noul) |
| Decoding | Autoregressive, token-by-token | Single forward pass, parallel |
| Latency (1 question, T4) | Hundreds of ms to seconds | ~33–40 ms [citation:2] |
| Hallucination risk | Yes (invents text) | No (bounded to defined options) |
| Calibration | Poor without post-hoc fitting | Trained with RLCD for calibrated confidence |
| Cost | Per-token API pricing | Free after local download |

### 3.5 Performance Benchmarks

On a Tesla T4 GPU, Laya's measured latency for a single question is **39.5 ms**
(English checkpoint) and **32.8 ms** (multilingual). When batching 10 questions, the
multilingual checkpoint reaches **7.2 ms per question** [citation:2]. For reference,
TypeSafe's Jev has been independently measured at 236–276 ms p50, making Laya
approximately **6–7× faster** for single-question decisions [citation:2].

On accuracy benchmarks, the routed Laya system scores comparably to Jev on
typed-decision tasks (0.766 vs 0.727 on a 2,000-decision benchmark) [citation:2].

---

## 4. The Dataset: Mixed App Store & Yelp Reviews

### 4.1 Source and Composition

The dataset used is **`mixed_app_yelp_dataset.csv`**, a combined corpus of:

- **App Store reviews**: short, informal user feedback about mobile applications
  (e.g., "this app is pathetic", "wow this app is a awesome").
- **Yelp reviews**: longer, more detailed restaurant and business reviews
  (e.g., "as we first walked into the restaurant the hostess was having a sidebar
  conversation…").

The mixing of these two sources creates a challenging classification task: the model
must handle both very short texts (sometimes just 2–3 words) and longer narrative
reviews, across informal spelling, slang, and domain-specific vocabulary.

### 4.2 Size and Labels

| Property | Value |
|---|---|
| Total rows | 215,760 |
| Columns | `text`, `label` |
| Label values | `0` (Negative), `1` (Positive) |
| Training split (80%) | 172,608 rows |
| Test split (20%) | 43,152 rows |

Label distribution in the training set after the internal train/validation split:
**Negative: 82,701**, **Positive: 72,647**. Class weights were applied to compensate
for the imbalance: `{'Negative': 0.97, 'Positive': 1.03}`.

### 4.3 Splitting Strategy

A critical design decision: the dataset was split **before** any training began. An
80/20 stratified split was performed on the full CSV, producing:

- `df_train` (172,608 rows): used to generate the training JSONL.
- `df_test` (43,152 rows): saved separately as `_test_split.csv`, never seen by the
  model during training or validation.

This prevents **data leakage**: the evaluation metrics reflect true held-out
performance, not memorization.

The training script then further split `df_train` internally into 80/10/10 for
train/calibration/validation [citation:1], resulting in:

- Train: 155,348 rows.
- Validation: 17,260 rows.

### 4.4 Data Characteristics

The reviews exhibit several properties that make sentiment classification non-trivial:

- **Informal language**: lowercase, missing punctuation, slang ("lurv", "ahh", "bro").
- **Mixed sentiment**: some reviews contain both positive and negative elements
  (e.g., "i thing it is bad and good at that same point of time").
- **Very short texts**: some reviews are only 2–4 words ("good nice wow", "bad").
- **Negation**: phrases like "not bad" (positive) vs. "not good" (negative) require
  compositional understanding.

The model performed well on these cases, as shown in the qualitative evaluation
(Section 7).

---

## 5. Converting the Dataset to Laya JSONL Format

Laya fine-tuning expects a **JSONL** file where each line is a JSON object. The
structure must match what the training script's `load_rows` function expects.

### 5.1 The Correct JSONL Schema

Each line must contain:

```json
{
  "state": {"text": "<review text>"},
  "questions": [{
    "id": "sentiment",
    "type": "choice",
    "instructions": "What is the sentiment of this review?",
    "criteria": ["Negative", "Positive"]
  }],
  "label": "Positive"
}