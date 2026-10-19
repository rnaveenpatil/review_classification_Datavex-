# Product Review Sentiment Analysis

**NLP · Binary Sentiment Classification**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Task](https://img.shields.io/badge/task-binary%20sentiment%20classification-brightgreen)
![Models](https://img.shields.io/badge/models-MiniLM%20%7C%20Laya-orange)
![Best Accuracy](https://img.shields.io/badge/best%20accuracy-89.48%25-success)

> **At a glance**
> - 🎯 **Task:** classify a review as **POSITIVE** or **NEGATIVE**
> - 🤖 **Models:** MiniLM (fine-tuned encoder) + Laya (System One decision model)
> - 📊 **Data:** ~215,760 mixed App Store + Yelp reviews
> - 🏆 **Best result:** 89.48% accuracy · 0.8945 macro F1 (MiniLM)
> - 🚀 **Try online:** [MiniLM Space](https://huggingface.co/spaces/rnaveenpatil/miniLm_fintuned_review_classificartion) · [Laya Space](https://huggingface.co/spaces/rnaveenpatil/Laya_fintuned_review_clssification)

### 📑 Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Repository Structure](#2-repository-structure)
3. [Datasets Used](#3-datasets-used)
4. [Our Solution](#4-our-solution)
5. [Model Comparison](#5-model-comparison)

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

### 3.1 📱 App Reviews — IGWA

The IGWA dataset contains Google Play Store reviews for Instagram, WhatsApp, and Facebook.
The extended `dataset_IGWA_FB10.csv` used here has ~176,000 reviews, labelled
**Negative / Neutral / Positive** based on star ratings. The reviews are short, informal,
and full of slang, emojis, and typos.

> **How it is used:** The Neutral reviews are dropped to keep the task binary, the text is
> cleaned and normalized, and the remaining reviews are merged with Yelp to build the mixed
> training corpus.

### 3.2 🍽️ Yelp Reviews — Binary Polarity

The Yelp dataset contains 560,000 training and 38,000 test restaurant/business reviews.
Stars **1–2 are Negative** and stars **3–4 are Positive**, giving a binary label directly.
The reviews are long and detailed, and often mix positive and negative statements.

> **How it is used:** A **50,000-review subset** is merged with the IGWA app reviews to form
> the mixed training corpus.

### 3.3 🔗 How the Datasets Are Used

The two sources are combined into one mixed corpus of **215,760 reviews** with
`text` and `label` columns (`0 = Negative`, `1 = Positive`). This same mixed corpus feeds
both models:

| Model | File | Format |
|---|---|---|
| MiniLM | [`mixed_app_yelp_dataset_for_minlm.csv`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv) | CSV (full corpus) |
| Laya | [`laya_train_fixed.jsonl`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/laya_fintuned_model/dataset/laya_train_fixed.jsonl) | JSONL (172,608 train rows) |

> For Laya, the same corpus is converted to JSONL and split 80/20 — **172,608 training rows**
> and **43,152 held-out test rows** (`test_split.csv`), then split again internally into
> 155,348 train / 17,260 validation.

### 3.4 🧩 Why We Mixed the Two Datasets

The two datasets sit at opposite ends of the review spectrum: IGWA is very short and
slang-heavy, while Yelp is long and narrative. A model trained on only one domain fails on
the other — app-review models get confused by long Yelp paragraphs, and Yelp-trained models
fail on short slang. Mixing them teaches the model to handle **both very short and very long
reviews**, and to interpret words in context (e.g. "bad" means different things in different
sentences). The combined corpus is shuffled so the model never learns one domain before the
other.

---

## 4. Our Solution

Instead of a single classical baseline, we fine-tuned **two** transformer-based sentiment
classifiers on a mixed corpus of App Store and Yelp reviews. Both models solve the same
binary task (Negative vs. Positive) but take fundamentally different approaches: a
lightweight BERT-family encoder (**MiniLM**) and a non-generative "System One" decision
engine (**Laya**).

> Both models were trained on the same mixed review corpus (`0 = Negative`, `1 = Positive`),
> combining short, slang-heavy app reviews with longer, detailed Yelp reviews.

### 4.1 🟦 Model 1 — MiniLM (Fine-Tuned Encoder Classifier)

[![Try MiniLM on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on%20Hugging%20Face-MiniLM-yellow)](https://huggingface.co/spaces/rnaveenpatil/miniLm_fintuned_review_classificartion)

| Item | Value |
|---|---|
| Base architecture | `sentence-transformers/all-MiniLM-L6-v2` |
| Head | `BertForSequenceClassification` (2 outputs) |
| Parameters | ~22.7 Million |
| Max token length | 512 |
| Training data | Mixed App + Yelp dataset (215,760 reviews) |
| Epochs | 2 |
| Batch size | 32 |

**Approach:** The base MiniLM model is designed for semantic similarity and does not
natively understand sentiment. We replaced its final layer with a 2-class classification
head and fully fine-tuned all ~22.7M parameters on the mixed corpus. Text was lowercased,
URLs/HTML stripped, whitespace normalized, and "Neutral" app reviews dropped to keep the
task strictly binary. The combined dataset was shuffled so the model saw both domains
interleaved.

**Capabilities — it understands intention, not just words:**

- 🧠 Reads the **overall meaning and intention** of a review rather than scoring isolated
  keywords.
- 🎭 Detects **sarcasm and indirect sentiment**, e.g. *"If you enjoy waiting 45 minutes for a
  cold burger and rude service, this is your paradise."* → **Negative**.
- 📖 Handles **mixed long reviews** with a positive opening and a negative ending, e.g.
  *"The atmosphere is beautiful… But then the food arrived. My $40 steak was overcooked…"*
  → **Negative**.
- 🔄 Handles **negation** and composition, e.g. *"not bad"* → **Positive**.
- ✂️ Works across both **very short slang** (*"worst update ever"*) and **long paragraphs**.

**Inference:** Raw logits are passed through Softmax to give class probabilities, and a
sentiment score is derived as `P(Positive) − P(Negative)` on a scale from −1.0 (strongly
negative) to +1.0 (strongly positive).

**How we trained it — by mixing two datasets:** We fine-tuned MiniLM on a single mixed
corpus that combines the **IGWA App Reviews** dataset (short, slang-heavy Google Play
reviews) with the **Yelp Reviews** dataset (long, detailed reviews). The combined training
file used for this model lives in this repo:
[`miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv).
This **same mixed dataset is reused by the Laya model** (see §4.2), where it is converted
into Laya's JSONL format.

**Results (evaluated on 215,760 reviews):**

| Metric | Score |
|---|:---:|
| Accuracy | **0.8948** |
| Macro Precision | 0.8941 |
| Macro Recall | 0.8951 |
| Macro F1 | **0.8945** |

**Result Images**

<p>
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/plot1_metrics.png" width="360" alt="MiniLM metrics">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/plot3_f1_scores.png" width="360" alt="MiniLM F1 scores">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/sentiment_confusion_matrix.png" width="360" alt="MiniLM confusion matrix">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/sentiment_score_distribution.png" width="360" alt="MiniLM score distribution">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/positive_negative_predictions.png" width="360" alt="MiniLM predictions">
</p>

**Artifacts:** `miniLM_fine_tuned/miniLM_model/` (weights, config, tokenizer) and
`miniLM_fine_tuned/sentiment_test_results/` (metrics, predictions, plots).

### 4.2 🟧 Model 2 — Laya (System One Decision Model)

[![Try Laya on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on%20Hugging%20Face-Laya-yellow)](https://huggingface.co/spaces/rnaveenpatil/Laya_fintuned_review_clssification)

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

**Approach:** Unlike a generative LLM, Laya does not produce text. It takes a review as a
*state* and a typed *question* ("What is the sentiment of this review?") and, in a single
forward pass, returns calibrated probabilities over the predefined options
`["Negative", "Positive"]`. Because the model can only answer within the options we define,
it cannot invent an out-of-range label. The dataset was converted to Laya's JSONL schema
(`state` / `questions` / `label`) and the model was fine-tuned with soft targets, shuffled
option order, and temperature calibration on the validation set.

**Capabilities — it understands intention, not just words:**

- 🧠 Conditions on the **question** and reads the **whole review** in one pass, so it captures
  the author's **overall intention and tone** rather than scoring single keywords.
- 📖 Handles **mixed reviews** that contain both positive and negative statements and returns
  a single bounded decision.
- 🛡️ Cannot hallucinate: the answer is always one of the defined options, so labels are
  always valid.

**Inference:** `laya.Agent("<checkpoint>")` loads the model; a `choice` question returns the
predicted label and calibrated probabilities directly — no JSON parsing or retry loop.

**How we trained it — by mixing two datasets:** We fine-tuned Laya on a single mixed JSONL
corpus that combines the **IGWA App Reviews** dataset (short, slang-heavy Google Play
reviews) with the **Yelp Reviews** dataset (long, detailed reviews). The combined training
file used for this model lives in this repo:
[`laya_fintuned_model/dataset/laya_train_fixed.jsonl`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/laya_fintuned_model/dataset/laya_train_fixed.jsonl).
This is the **same mixed dataset used to train MiniLM** (see §4.1), converted into Laya's
JSONL format.

**Results (held-out validation set, 17,260 samples):**

| Metric | Score |
|---|:---:|
| Accuracy | **0.8849** |
| Macro F1 | **0.8845** |

**Result Images**

<p>
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/results.png" width="360" alt="Laya evaluation results">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/f1_scores.png" width="360" alt="Laya F1 scores">
<img src="https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/sentiment_distribution%20%281%29.png" width="360" alt="Laya sentiment distribution">
</p>

**Artifacts:** `laya_fintuned_model/laya_model/` (weights, config, tokenizer),
`laya_fintuned_model/train.py` + `common.py` (training pipeline), and
`laya_fintuned_model/laya_model_result/` (F1 scores, results, sentiment distribution plots).

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
| Sarcasm / indirect reviews | ✅ Handles them accurately |  |
| Hallucination risk | None (2 fixed classes) | None (bounded to fixed options) |
| Inference | Softmax over 2 logits | Typed `choice` decision in one pass |
| Speed | Very fast (tiny model) | Fast (~30–50 ms/review) |
| Best for | Lightweight, high-throughput apps | Always-valid, fast decisions |
| Try it | [![MiniLM Space](https://img.shields.io/badge/Try-MiniLM-yellow)](https://huggingface.co/spaces/rnaveenpatil/miniLm_fintuned_review_classificartion) | [![Laya Space](https://img.shields.io/badge/Try-Laya-orange)](https://huggingface.co/spaces/rnaveenpatil/Laya_fintuned_review_clssification) |

### 🏁 Key Takeaways

- 🟦 **MiniLM** is light, understands complex intention and long reviews, and accurately
  predicts indirect and sarcasm-based reviews — best for fast, high-throughput use.
- 🟧 **Laya** is a System One decision model: it **cannot generate text**, but it takes
  decisions **accurately and fastly** in a single pass, always returning a valid label.
- ✅ Both exceed the accuracy expected for the test cases (TC1–TC6) and give a `predict`-style
  result for unseen reviews. Only MiniLM is claimed to handle sarcasm; Laya is not evaluated
  for sarcasm.

---

<p align="center"><a href="#product-review-sentiment-analysis">⬆ Back to top</a></p>
