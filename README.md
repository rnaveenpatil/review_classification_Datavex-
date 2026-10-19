# Product Review Sentiment Analysis

**NLP · Binary Sentiment Classification**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Task](https://img.shields.io/badge/task-binary%20sentiment%20classification-brightgreen)
![Models](https://img.shields.io/badge/models-MiniLM%20%7C%20Laya-orange)
![Best Accuracy](https://img.shields.io/badge/best%20accuracy-89.48%25-success)

---

> ### 🌟 What we built, in simple words
>
> We built an application that reads a review and tells you whether it is **POSITIVE** or
> **NEGATIVE**.
>
> We trained **two different AI models** on a mix of app-store and Yelp reviews, compared
> them, and **deployed both online** so anyone can try them in a browser — no setup needed.

### 🔎 At a Glance

| Question | Answer |
|---|---|
| **What does it do?** | Classifies a review as POSITIVE or NEGATIVE |
| **How many models?** | Two — **MiniLM** and **Laya** |
| **What data was used?** | 215,760 mixed App Store + Yelp reviews |
| **Best accuracy?** | **89.48%** (MiniLM) |
| **Can I try it online?** | Yes — live demo links are at the end of each model section |

### 📑 Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Repository Structure](#2-repository-structure)
3. [Datasets Used](#3-datasets-used)
4. [Our Solution](#4-our-solution)
5. [Model Comparison](#5-model-comparison)
6. [Glossary](#6-glossary)

---

## 1. Problem Statement

Create an application that determines whether a product review expresses a **POSITIVE** or **NEGATIVE** sentiment. The system must use a labelled review dataset to train a model and must support predictions on new reviews.

### 1.1 Requirements

1. Load a dataset with `review_text` and `sentiment` columns.
2. Normalize text and handle empty reviews.
3. Train a text classifier using TF-IDF and Logistic Regression, Naive Bayes, or a similar method.
4. Evaluate performance using accuracy, precision, recall, and F1-score.
5. Accept a new review and return its predicted sentiment.
6. Demonstrate predictions on reviews not used for training.

### 1.2 Test Cases

| Test case | Input | Expected behaviour |
|:---:|---|---|
| **TC1** | "The product is excellent and works perfectly" | POSITIVE, provided the model has learned similar examples |
| **TC2** | "The product broke after one day" | NEGATIVE, provided the model has learned similar examples |
| **TC3** | Empty review | Reject or return a clear validation message |
| **TC4** | Review containing emojis and punctuation | Process without crashing |
| **TC5** | A completely new review | Return a prediction |
| **TC6** | Review containing both positive and negative statements | Return the model's predicted sentiment without claiming perfect understanding of context |

> ✅ **Acceptance criteria:** A working sentiment classifier, evaluation metrics, and a function for predicting sentiment on unseen reviews.

---

## 2. Repository Structure

Below is the folder layout. A one-line explanation of every file is in the collapsible list.

```text
review_classification_Datavex-/
├── README.md
├── Problem Statement - Naveen Kumar.pdf
├── .gitattributes
├── original dataset/
│   ├── app review dataset/
│   │   ├── README.md
│   │   ├── dataset_IGWA_FB10.csv
│   │   ├── dataset_IGWA_only.csv
│   │   ├── logreg_*.joblib
│   │   └── plots/
│   │       ├── metrics_*.json
│   │       ├── confusion_*.png
│   │       └── score_dist_*.png
│   └── yelp_dataset/
│       ├── README.md
│       ├── train.csv
│       ├── test.csv
│       └── .cache/huggingface/
├── miniLM_fine_tuned/
│   ├── readme.md
│   ├── miniLM_model/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── training_args.bin
│   ├── dataset/
│   │   └── mixed_app_yelp_dataset_for_minlm.csv
│   └── sentiment_test_results/
│       ├── sentiment_metrics.csv
│       ├── sentiment_predictions.csv
│       └── *.png
└── laya_fintuned_model/
    ├── readme.md
    ├── train.py
    ├── common.py
    ├── task.json
    ├── dataset/
    │   └── laya_train_fixed.jsonl
    ├── test_split.csv
    ├── laya_model/
    │   ├── model.safetensors
    │   ├── rl_agent_config.json
    │   ├── task.json
    │   ├── training_log.json
    │   ├── encoder/config.json
    │   └── tokenizer/
    └── laya_model_result/
        └── *.png
```

<details>
<summary><b>📄 Click to see what every file does</b></summary>

<br>

| Path | What it does (one line) |
|---|---|
| `README.md` | Main project documentation — problem statement, models, results, and comparison. |
| `Problem Statement - Naveen Kumar.pdf` | The original assignment brief with requirements and test cases. |
| `.gitattributes` | Configures Git LFS for large files (`.safetensors`, `.csv`, `.jsonl`, `.bin`). |
| `original dataset/app review dataset/README.md` | Dataset card describing the IGWA app-review dataset. |
| `original dataset/app review dataset/dataset_IGWA_FB10.csv` | Extended app-review dataset (~176,000 Google Play reviews, including Facebook). |
| `original dataset/app review dataset/dataset_IGWA_only.csv` | Instagram + WhatsApp-only app-review dataset. |
| `original dataset/app review dataset/logreg_*.joblib` | Pre-trained TF-IDF + Logistic Regression baseline models. |
| `original dataset/app review dataset/plots/metrics_*.json` | Accuracy, precision, recall, and F1 scores for the baseline models. |
| `original dataset/app review dataset/plots/confusion_*.png` | Confusion-matrix plots for the baseline models. |
| `original dataset/app review dataset/plots/score_dist_*.png` | Prediction score-distribution plots for the baseline models. |
| `original dataset/yelp_dataset/README.md` | Dataset card describing the Yelp polarity dataset. |
| `original dataset/yelp_dataset/train.csv` | Yelp training reviews with binary sentiment labels. |
| `original dataset/yelp_dataset/test.csv` | Yelp held-out test reviews. |
| `original dataset/yelp_dataset/.cache/huggingface/` | Hugging Face download cache for the Yelp files. |
| `miniLM_fine_tuned/readme.md` | Documentation for the MiniLM model and training procedure. |
| `miniLM_fine_tuned/miniLM_model/config.json` | MiniLM model architecture and hyperparameter configuration. |
| `miniLM_fine_tuned/miniLM_model/model.safetensors` | Fine-tuned MiniLM model weights (~90 MB). |
| `miniLM_fine_tuned/miniLM_model/tokenizer.json` | Tokenizer vocabulary used to encode review text. |
| `miniLM_fine_tuned/miniLM_model/tokenizer_config.json` | Tokenizer settings (special tokens, padding, truncation). |
| `miniLM_fine_tuned/miniLM_model/training_args.bin` | Saved Hugging Face `TrainingArguments` from training. |
| `miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv` | Mixed App + Yelp dataset (215,760 rows) used to train MiniLM. |
| `miniLM_fine_tuned/sentiment_test_results/sentiment_metrics.csv` | Final accuracy, precision, recall, and F1 for MiniLM. |
| `miniLM_fine_tuned/sentiment_test_results/sentiment_predictions.csv` | Per-review predictions produced by MiniLM on the test set. |
| `miniLM_fine_tuned/sentiment_test_results/*.png` | Evaluation plots (metrics, F1, confusion matrix, score distribution). |
| `laya_fintuned_model/readme.md` | Documentation for the Laya model and training procedure. |
| `laya_fintuned_model/train.py` | Fine-tuning script for the Laya System One decision model. |
| `laya_fintuned_model/common.py` | Shared task/data loading and validation helpers. |
| `laya_fintuned_model/task.json` | Defines the labels and the sentiment `choice` question. |
| `laya_fintuned_model/dataset/laya_train_fixed.jsonl` | Training data in Laya's JSONL format (172,608 rows). |
| `laya_fintuned_model/test_split.csv` | Held-out test split for evaluating Laya (43,152 rows). |
| `laya_fintuned_model/laya_model/model.safetensors` | Fine-tuned Laya model weights (~1.2 GB). |
| `laya_fintuned_model/laya_model/rl_agent_config.json` | Laya checkpoint config (encoder, temperatures, fine-tuning info). |
| `laya_fintuned_model/laya_model/task.json` | Label/question definition bundled with the checkpoint. |
| `laya_fintuned_model/laya_model/training_log.json` | Training history, validation metrics, and fitted temperature. |
| `laya_fintuned_model/laya_model/encoder/config.json` | Encoder (mmBERT-base) architecture configuration. |
| `laya_fintuned_model/laya_model/tokenizer/` | Tokenizer files used by the Laya encoder. |
| `laya_fintuned_model/laya_model_result/*.png` | Result plots (F1 scores, evaluation results, sentiment distribution). |

</details>

---

## 3. Datasets Used

We used **two public datasets** from two very different review domains.

### 3.1 📱 App Reviews — IGWA

The IGWA dataset contains Google Play Store reviews for Instagram, WhatsApp, and Facebook.
The extended `dataset_IGWA_FB10.csv` used here has ~176,000 reviews, labelled
**Negative / Neutral / Positive** based on star ratings. The reviews are **short, informal,
and full of slang, emojis, and typos**.

> **How it is used:** We drop the Neutral reviews to keep the task binary, clean the text,
> and merge the remaining reviews with Yelp to build the mixed training corpus.

### 3.2 🍽️ Yelp Reviews — Binary Polarity

The Yelp dataset contains 560,000 training and 38,000 test restaurant/business reviews.
Stars **1–2 are Negative** and stars **3–4 are Positive**, giving a binary label directly.
The reviews are **long, detailed, and often mix positive and negative statements**.

> **How it is used:** A **50,000-review subset** is merged with the IGWA app reviews to form
> the mixed training corpus.

### 3.3 🔗 How the Two Datasets Were Mixed

The two sources are combined into one mixed corpus of **215,760 reviews** with `text` and
`label` columns (`0 = Negative`, `1 = Positive`).

| # | Dataset | Source file | What we take | Rows contributed |
|:---:|---|---|---|---:|
| 1 | 📱 App Reviews (IGWA) | [`dataset_IGWA_FB10.csv`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/original%20dataset/app%20review%20dataset/dataset_IGWA_FB10.csv) | Instagram / WhatsApp / Facebook reviews, **Neutral dropped** | ~166,000 |
| 2 | 🍽️ Yelp Reviews | [`train.csv`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/original%20dataset/yelp_dataset/train.csv) | Random **50,000-review subset** | 50,000 |
| 3 | 🧩 **Mixed corpus** | — | Combine the two, **shuffle**, relabel as binary | **215,760** |

This same mixed corpus then feeds both models:

| Model | File | Format |
|---|---|---|
| MiniLM | [`mixed_app_yelp_dataset_for_minlm.csv`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv) | CSV (full corpus) |
| Laya | [`laya_train_fixed.jsonl`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/laya_fintuned_model/dataset/laya_train_fixed.jsonl) | JSONL (172,608 train rows) |

> For Laya, the same corpus is converted to JSONL and split 80/20 — **172,608 training rows**
> and **43,152 held-out test rows**, then split again internally into 155,348 train / 17,260
> validation.

### 3.4 🧩 Why We Mixed Them

The two datasets sit at opposite ends of the review spectrum:

| Dataset | Review style | Why it is needed |
|---|---|---|
| 📱 IGWA | Short, slangy, emoji-heavy | Teaches the model to handle real app-store language |
| 🍽️ Yelp | Long, detailed, narrative | Teaches the model to follow context across many sentences |

A model trained on only one domain fails on the other. Mixing them teaches the model to
handle **both very short and very long reviews**, and to read words in context (e.g. "bad"
can mean different things in different sentences). The combined corpus is shuffled so the
model never learns one domain before the other.

---

## 4. Our Solution

> **In simple words:** We did not build a model from scratch. We took **two pre-trained
> models that already understand language** and **fine-tuned** them on our labelled reviews,
> so they learned this specific POSITIVE vs. NEGATIVE task.

1. 🟦 **MiniLM** — a small, fast BERT-family encoder. We added a new 2-class classification
   head and fine-tuned it on the mixed corpus.
2. 🟧 **Laya** — a non-generative "System One" decision model. We fine-tuned it to answer a
   `choice` question: `Negative` or `Positive`.

Both models solve the same task but in different ways. We trained, evaluated, and then
**deployed both as live demos on Hugging Face Spaces**.

### 🧭 How It Works — Step by Step

1. **Collect** app-store reviews and Yelp reviews.
2. **Clean** the text and drop Neutral app reviews so the task is binary.
3. **Mix and shuffle** both sources into one corpus of 215,760 reviews.
4. **Fine-tune two models** on the same corpus: MiniLM and Laya.
5. **Evaluate** with accuracy, precision, recall, and F1-score.
6. **Deploy** both models on Hugging Face Spaces for live testing.

```text
📱 App Reviews ─┐
                ├──▶ 🧹 Clean + Mix ──▶ 🧩 215,760 reviews
🍽️ Yelp Reviews ┘                              │
                                               ├──▶ 🟦 Fine-tune MiniLM ──▶ 89.48% ──▶ 🚀 Online demo
                                               └──▶ 🟧 Fine-tune Laya   ──▶ 88.49% ──▶ 🚀 Online demo
```

### 4.1 🟦 Model 1 — MiniLM (Fine-Tuned Encoder Classifier)

> **In simple words:** A small and fast AI model that reads a review and scores whether it
> is positive or negative. Because it is lightweight, it is cheap to host and quick to run.

| Item | Value |
|---|---|
| Base architecture | `sentence-transformers/all-MiniLM-L6-v2` |
| Head | `BertForSequenceClassification` (2 outputs) |
| Parameters | ~22.7 Million |
| Max token length | 512 |
| Training data | Mixed App + Yelp dataset (215,760 reviews) |
| Epochs | 2 |
| Batch size | 32 |

**How it was trained:** The base MiniLM model is designed for semantic similarity, not
sentiment. We replaced its final layer with a 2-class classification head and fully
fine-tuned all ~22.7M parameters on the mixed corpus. Text was lowercased, URLs/HTML were
removed, whitespace was normalized, and Neutral app reviews were dropped. The dataset was
shuffled so the model saw both domains interleaved.

**What it can do:**

- 🧠 Reads the **overall meaning and intention** of a review rather than scoring isolated
  keywords.
- 🎭 Captures **some level of sarcasm and indirect sentiment** (not perfectly — tricky
  context can still be missed), e.g. *"If you enjoy waiting 45 minutes for a cold burger and
  rude service, this is your paradise."* → **Negative**.
- 📖 Handles **mixed long reviews** with a positive opening and a negative ending, e.g.
  *"The atmosphere is beautiful… But then the food arrived. My $40 steak was overcooked…"*
  → **Negative**.
- 🔄 Handles **negation**, e.g. *"not bad"* → **Positive**.
- ✂️ Works on both **very short slang** (*"worst update ever"*) and **long paragraphs**.

**How predictions are made:** The model outputs two scores (Negative, Positive). A Softmax
turns them into probabilities, and the sentiment score is `P(Positive) − P(Negative)`, from
−1.0 (strongly negative) to +1.0 (strongly positive).

**How the training data was built:** We fine-tuned MiniLM on the mixed corpus described in
§3.3 — **IGWA App Reviews** (short and slang-heavy) plus **Yelp Reviews** (long and
detailed). The file is in this repo:
[`miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv).
The **same mixed dataset is reused by Laya** (§4.2), converted to JSONL.

**Results — evaluated on 215,760 reviews:**

| Metric | What it means | Score |
|---|---|---|
| Accuracy | Share of reviews classified correctly | **0.8948** |
| Macro Precision | How often positive predictions are right | 0.8941 |
| Macro Recall | How many true positives are found | 0.8951 |
| Macro F1 | Balance of precision and recall | **0.8945** |

**Result Images**

<p>
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/plot1_metrics.png" width="360" alt="MiniLM metrics">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/plot3_f1_scores.png" width="360" alt="MiniLM F1 scores">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/sentiment_confusion_matrix.png" width="360" alt="MiniLM confusion matrix">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/sentiment_score_distribution.png" width="360" alt="MiniLM score distribution">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/positive_negative_predictions.png" width="360" alt="MiniLM predictions">
</p>

**Files:** `miniLM_fine_tuned/miniLM_model/` (weights, config, tokenizer) and
`miniLM_fine_tuned/sentiment_test_results/` (metrics, predictions, plots).

> 🚀 **LIVE DEMO — Try MiniLM now!**
>
> We **fine-tuned and deployed this model on Hugging Face Spaces**.
>
> **➡️ [Click here to open the MiniLM Review Sentiment Classifier deployed on huggingface space](https://huggingface.co/spaces/rnaveenpatil/miniLm_fintuned_review_classificartion)**

### 4.2 🟧 Model 2 — Laya (System One Decision Model)

> **In simple words:** Laya is a **decision model, not a chatbot**. It does not write text.
> It reads the review and simply picks one answer: **Negative** or **Positive** — quickly
> and with a calibrated confidence.

| Item | Value |
|---|---|
| Base model | `convaiinnovations/laya` — multilingual (mmBERT-base encoder) |
| Model type | Non-autoregressive "System One" decision engine |
| Parameters | ~322 Million (base) |
| Decision type | `choice` — options `["Negative", "Positive"]` |
| Max token length | 1,024 |
| Training data | 155,348 rows (90/10 train/validation split) |
| Validation rows | 17,260 |
| Epochs | 1 |
| Batch size | 8 |

**How it works:** Unlike a generative LLM, Laya does not produce text. It takes a review as
a *state* and a typed *question* ("What is the sentiment of this review?") and, in a single
forward pass, returns calibrated probabilities for the predefined options
`["Negative", "Positive"]`. Because the answer must be one of those options, the model
cannot invent an out-of-range label. The data was converted to Laya's JSONL schema
(`state` / `questions` / `label`) and fine-tuned with soft targets, shuffled option order,
and temperature calibration.

**What it can do:**

- 🧠 Conditions on the **question** and reads the **whole review** in one pass, so it captures
  the author's **overall intention and tone** rather than single keywords.
- 📖 Handles **mixed reviews** that contain both positive and negative statements and returns
  one bounded decision.
- 🛡️ Cannot hallucinate: the answer is always one of the defined options, so labels are
  always valid.

**How predictions are made:** `laya.Agent("<checkpoint>")` loads the model. A `choice`
question returns the predicted label and calibrated probabilities directly — no JSON to
parse and no retry loop.

**How the training data was built:** We fine-tuned Laya on the **same mixed corpus** used
for MiniLM, converted to Laya's JSONL format. The file is in this repo:
[`laya_fintuned_model/dataset/laya_train_fixed.jsonl`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/laya_fintuned_model/dataset/laya_train_fixed.jsonl).
It combines **IGWA App Reviews** (short, slang-heavy) with **Yelp Reviews** (long, detailed).

**Results — held-out validation set, 17,260 reviews:**

| Metric | What it means | Score |
|---|---|---|
| Accuracy | Share of reviews classified correctly | **0.8849** |
| Macro F1 | Balance of precision and recall | **0.8845** |

**Result Images**

<p>
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/results.png" width="360" alt="Laya evaluation results">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/f1_scores.png" width="360" alt="Laya F1 scores">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/sentiment_distribution%20%281%29.png" width="360" alt="Laya sentiment distribution">
</p>

**Files:** `laya_fintuned_model/laya_model/` (weights, config, tokenizer),
`laya_fintuned_model/train.py` + `common.py` (training pipeline), and
`laya_fintuned_model/laya_model_result/` (F1 scores, results, sentiment distribution plots).

> 🚀 **LIVE DEMO — Try Laya now!**
>
> We **fine-tuned and deployed this model on Hugging Face Spaces**.
>
> **➡️ [Click here to open the Laya Review Sentiment Classifier deployed on huggingface space](https://huggingface.co/spaces/rnaveenpatil/Laya_fintuned_review_clssification)**

---

## 5. Model Comparison

### ⚖️ MiniLM vs. Laya

| Point | 🟦 MiniLM (fine-tuned encoder) | 🟧 Laya (System One) |
|---|---|---|
| Base model | `all-MiniLM-L6-v2` (BERT family) | `convaiinnovations/laya` (mmBERT-base) |
| Type | Encoder + classification head | Non-generative System One decision model |
| Parameters | ~22.7 Million (lightweight) | ~322 Million |
| Size on disk | ~90 MB | ~1.2 GB |
| Generates text? | No | **No — it only takes decisions** |
| Max review length | 512 tokens | 1,024 tokens |
| Training data | Mixed App + Yelp (215,760 reviews) | Same mixed data, JSONL (172,608 train rows) |
| Training setup | 2 epochs · batch 32 · lr 2e-5 | 1 epoch · batch 8 · lr 2e-5 |
| Accuracy | **0.8948** | 0.8849 |
| Macro Precision | **0.8941** | — |
| Macro Recall | **0.8951** | — |
| Macro F1 | **0.8945** | 0.8845 |
| Sarcasm / indirect reviews | ✅ Captures some level of sarcasm | ➖ Not evaluated |
| Hallucination risk | None (2 fixed classes) | None (bounded to fixed options) |
| Inference | Softmax over 2 logits | Typed `choice` decision in one pass |
| Speed | Very fast (tiny model) | Fast (~30–50 ms/review) |
| Best for | Lightweight, high-throughput apps | Always-valid, fast decisions |
| Try it | [![MiniLM Space](https://img.shields.io/badge/Try-MiniLM-yellow)](https://huggingface.co/spaces/rnaveenpatil/miniLm_fintuned_review_classificartion) | [![Laya Space](https://img.shields.io/badge/Try-Laya-orange)](https://huggingface.co/spaces/rnaveenpatil/Laya_fintuned_review_clssification) |

### 🏁 Key Takeaways

- 🟦 **MiniLM** is light, understands complex intention and long reviews, and captures
  **some level of sarcasm** and indirect sentiment — best for fast, high-throughput use.
- 🟧 **Laya** is a System One decision model: it **cannot generate text**, but it takes
  decisions **accurately and fastly** in a single pass, always returning a valid label.
- ✅ Both exceed the accuracy expected for the test cases (TC1–TC6) and give a `predict`-style
  result for unseen reviews. Only MiniLM is claimed to handle some level of sarcasm; Laya is
  not evaluated for sarcasm.

---

## 6. Glossary

Simple definitions for terms used in this README:

| Term | Simple meaning |
|---|---|
| **Sentiment** | The feeling of a review — positive or negative. |
| **Fine-tuning** | Taking a model that already understands language and teaching it one specific task with labelled examples. |
| **Encoder** | The part of a model that reads text and turns it into numbers that capture meaning. |
| **System One** | A model that makes a fast, bounded decision instead of writing free text. |
| **Token** | A small piece of text (a word or part of a word) that the model reads. |
| **Accuracy** | The share of predictions that are correct. |
| **Precision** | Of the reviews predicted positive, how many really are positive. |
| **Recall** | Of the reviews that are actually positive, how many the model found. |
| **F1-score** | One balanced score that combines precision and recall (higher is better). |

---

<p align="center"><a href="#product-review-sentiment-analysis">⬆ Back to top</a></p>
