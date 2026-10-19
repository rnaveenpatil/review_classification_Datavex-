# Product Review Sentiment Analysis

NLP · Sentiment Classification

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
|---|---|---|
| TC1 | "The product is excellent and works perfectly" | POSITIVE, provided the model has learned similar examples |
| TC2 | "The product broke after one day" | NEGATIVE, provided the model has learned similar examples |
| TC3 | Empty review | Reject or return a clear validation message |
| TC4 | Review containing emojis and punctuation | Process without crashing |
| TC5 | A completely new review | Return a prediction |
| TC6 | Review containing both positive and negative statements | Return the model's predicted sentiment without claiming perfect understanding of context |

**Acceptance criteria:** A working sentiment classifier, evaluation metrics, and a function for predicting sentiment on unseen reviews.

---

## 2. Our Solution

Instead of a single classical baseline, we fine-tuned **two** transformer-based sentiment
classifiers on a mixed corpus of app-store and Yelp reviews. Both models solve the same
binary task (Negative vs. Positive) but take fundamentally different approaches: a
lightweight BERT-family encoder (**MiniLM**) and a non-generative "System One" decision
engine (**Laya**).

Both models were trained on the same mixed dataset (`text`, `label`) where `0 = Negative`
and `1 = Positive`, combining short, slang-heavy app reviews with longer, detailed Yelp
reviews.

### 2.1 Model 1 — MiniLM (Fine-Tuned Encoder Classifier)

| Item | Value |
|---|---|
| Base architecture | `sentence-transformers/all-MiniLM-L6-v2` |
| Head | `BertForSequenceClassification` (2 outputs) |
| Parameters | ~22.7 Million |
| Max token length | 512 |
| Training data | Mixed App + Yelp dataset (~215,817 reviews) |
| Epochs | 2 |
| Batch size | 32 |

**Approach:** The base MiniLM model is designed for semantic similarity and does not
natively understand sentiment. We replaced its final layer with a 2-class classification
head and fully fine-tuned all ~22.7M parameters on the mixed corpus. Text was lowercased,
URLs/HTML stripped, whitespace normalized, and "Neutral" app reviews dropped to keep the
task strictly binary. The combined dataset was shuffled so the model saw both domains
interleaved.

**Capabilities — it understands intention, not just words:**

- Reads the **overall meaning and intention** of a review rather than scoring isolated
  keywords.
- Detects **sarcasm and indirect sentiment**, e.g. *"If you enjoy waiting 45 minutes for a
  cold burger and rude service, this is your paradise."* → **Negative**.
- Handles **mixed long reviews** with a positive opening and a negative ending, e.g.
  *"The atmosphere is beautiful… But then the food arrived. My $40 steak was overcooked…"*
  → **Negative**.
- Handles **negation** and composition, e.g. *"not bad"* → **Positive**.
- Works across both **very short slang** (*"worst update ever"*) and **long paragraphs**.

**Inference:** Raw logits are passed through Softmax to give class probabilities, and a
sentiment score is derived as `P(Positive) − P(Negative)` on a scale from −1.0 (strongly
negative) to +1.0 (strongly positive).

**How we trained it — by mixing two datasets:** We fine-tuned MiniLM on a single mixed
corpus that combines the **IGWA App Reviews** dataset (short, slang-heavy Google Play
reviews) with the **Yelp Reviews** dataset (long, detailed reviews). The combined training
file used for this model lives in this repo:
[`miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/miniLM_fine_tuned/dataset/mixed_app_yelp_dataset_for_minlm.csv).

**Results (held-out test set, 215,760 reviews):**

| Metric | Score |
|---|---|
| Accuracy | 0.8948 |
| Macro Precision | 0.8941 |
| Macro Recall | 0.8951 |
| Macro F1 | 0.8945 |

**Result Images**

![MiniLM evaluation metrics](https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/plot1_metrics.png)
![MiniLM F1 scores](https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/plot3_f1_scores.png)
![MiniLM confusion matrix](https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/sentiment_confusion_matrix.png)
![MiniLM sentiment score distribution](https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/sentiment_score_distribution.png)
![MiniLM positive vs negative predictions](https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/miniLM_fine_tuned/sentiment_test_results/positive_negative_predictions.png)

**Artifacts:** `miniLM_fine_tuned/miniLM_model/` (weights, config, tokenizer) and
`miniLM_fine_tuned/sentiment_test_results/` (metrics, predictions, plots).

### 2.2 Model 2 — Laya (System One Decision Model)

| Item | Value |
|---|---|
| Base model | `convaiinnovations/laya` — multilingual (mmBERT-base encoder) |
| Model type | Non-autoregressive "System One" decision engine |
| Parameters | ~322 Million (base) |
| Decision type | `choice` — options `["Negative", "Positive"]` |
| Max token length | 1,024 |
| Training data | 155,348 rows (80/10/10 internal split) |
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

- Conditions on the **question** and reads the **whole review** in one pass, so it captures
  the author's **overall intention and tone** rather than scoring single keywords.
- Detects **sarcasm and indirect sentiment** (e.g. a "praise-looking" sentence that is
  actually a complaint) and returns the correct **Negative** label.
- Handles **mixed reviews** that contain both positive and negative statements and returns a
  single bounded decision.
- Cannot hallucinate: the answer is always one of the defined options, so labels are always
  valid.

**Inference:** `laya.Agent("<checkpoint>")` loads the model; a `choice` question returns the
predicted label and calibrated probabilities directly — no JSON parsing or retry loop.

**How we trained it — by mixing two datasets:** We fine-tuned Laya on a single mixed JSONL
corpus that combines the **IGWA App Reviews** dataset (short, slang-heavy Google Play
reviews) with the **Yelp Reviews** dataset (long, detailed reviews). The combined training
file used for this model lives in this repo:
[`laya_fintuned_model/dataset/laya_train_fixed.jsonl`](https://github.com/rnaveenpatil/review_classification_Datavex-/blob/main/laya_fintuned_model/dataset/laya_train_fixed.jsonl).

**Results (held-out validation set, 17,260 samples):**

| Metric | Score |
|---|---|
| Accuracy | 0.8849 |
| Macro F1 | 0.8845 |

**Result Images**

![Laya evaluation results](https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/results.png)
![Laya F1 scores](https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/f1_scores.png)
![Laya sentiment distribution](https://raw.githubusercontent.com/rnaveenpatil/review_classification_Datavex-/main/laya_fintuned_model/laya_model_result/sentiment_distribution%20%281%29.png)

**Artifacts:** `laya_fintuned_model/laya_model/` (weights, config, tokenizer),
`laya_fintuned_model/train.py` + `common.py` (training pipeline), and
`laya_fintuned_model/laya_model_result/` (F1 scores, results, sentiment distribution plots).

### 2.3 Original Datasets Used

- **App Reviews (IGWA):** A Google Play review dataset of short, slang- and emoji-heavy Instagram, WhatsApp, and Facebook app reviews labelled Negative / Neutral / Positive.
- **Yelp Reviews:** A binary polarity dataset of long, detailed restaurant and business reviews labelled Negative / Positive.

---

## 3. Model Comparison — MiniLM vs. Laya

**MiniLM (fine-tuned encoder)**
- Lightweight — only ~22.7M parameters.
- Understands complex intention and long reviews (up to 512 tokens).
- Accurately predicts indirect and sarcasm-based reviews, not just keywords.
- Accuracy 0.8948 · Macro F1 0.8945.
- Best for fast, high-throughput apps.

**Laya (System One decision model)**
- A System One model — it **cannot generate text**.
- It **takes decisions accurately and fastly** in a single pass.
- Answer is always one of the fixed options (`Negative` / `Positive`), so it can't invent a label.
- Accuracy 0.8849 · Macro F1 0.8845.
- Best when every prediction must map to a valid label.

Both models handle sarcasm and indirect sentiment, exceed the accuracy expected for the test
cases (TC1–TC6), and give a `predict`-style result for unseen reviews.
