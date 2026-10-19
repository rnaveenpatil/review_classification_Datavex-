# 🎯 Advanced Sentiment Analysis: Mixed-Domain Transformer

## 📌 Project Overview
This repository contains a robust, fine-tuned Natural Language Processing (NLP) model designed to classify the sentiment of product, app, and service reviews. 

Unlike standard sentiment models that fail when text is too long or too complex, this model was specifically engineered to handle **short, slang-heavy app reviews** as well as **long, detailed, psychologically manipulative (sarcastic/indirect) Yelp reviews**.

## 🤖 The Model
*   **Base Architecture:** `sentence-transformers/all-MiniLM-L6-v2` (A highly efficient, lightweight Transformer).
*   **Parameters:** ~22.7 Million.
*   **Task:** Binary Text Classification (Positive vs. Negative).
*   **Max Token Length:** 512 tokens (capable of processing long paragraphs).

### Why Fine-Tune this specific model?
The base MiniLM model is originally designed for semantic similarity (finding similar sentences). It does not natively know what "positive" or "negative" means. We took this base model, replaced its final layer, and added a new `BertForSequenceClassification` head with 2 outputs. We then **fully fine-tuned** the entire 22-million parameter network specifically for sentiment.

---

## 📊 The Datasets & Mixing Strategy

To make the model smart enough for any scenario, we merged two completely different text domains into one massive training set:

### 1. App Reviews Dataset (`dataset_IGWA_FB10.csv`)
*   **Size:** ~176,000 reviews.
*   **Characteristics:** Short, direct, full of slang, emojis, and typos (e.g., *"Worst update ever"*, *"Love it!!! 😍"*).
*   **Challenge:** Models trained only on this data fail on long texts because they just look for individual words and get confused by extra context.

### 2. Yelp Reviews Dataset (`Yelp_Reviews_for_Binary_Senti_Analysis`)
*   **Size:** 560,000 reviews (We used a 50,000 subset to keep training efficient).
*   **Characteristics:** Long, detailed restaurant reviews. These often contain mixed signals, polite openings, and negative endings (e.g., *"The hostess was sweet, but the food was horrible."*).
*   **Challenge:** Models trained only on this data fail on short app slang because they expect paragraphs.

### The Solution: The 215,817 Mixed Dataset
We combined the 176k App reviews with the 50k Yelp subset, resulting in **215,817 total reviews**. 
*   **Why mix?** By merging these datasets, the Transformer's "Self-Attention" mechanism learned to handle both short slang and long paragraphs. It learned that words like *"bad"* have different meanings depending on context.
*   **Preprocessing:** 
    *   Labels were normalized (0 = Negative, 1 = Positive).
    *   "Neutral" app reviews were dropped to maintain strict binary classification.
    *   Text was lowercased, URLs/HTML tags were stripped, and whitespace was normalized.
    *   The combined dataset was shuffled so the model didn't train on Yelp first and App reviews second.

---

## ⚙️ The Fine-Tuning Procedure

We used PyTorch and the Hugging Face `Trainer` API. 

1.  **Tokenization:** Every review was tokenized with `padding="max_length"` and `truncation=True`. Crucially, we set **`max_length=512`** to capture the full context of long Yelp reviews.
2.  **Training Setup:**
    *   **Epochs:** 2.
    *   **Batch Size:** 32.
    *   **Learning Rate:** 2e-5.
    *   **Optimizer:** AdamW.
    *   **Hardware:** NVIDIA T4 GPU.
3.  **Results:** The model achieved **~88.3% Accuracy** and a **Macro F1-Score of 0.8834** on the unseen validation set.

---

## 📈 Evaluation & Deployment

### The Final Output
The model outputs raw logits, which are passed through a Softmax function to get probabilities (e.g., Positive: 0.98, Negative: 0.02). 
*   The predicted label is the class with the highest probability.
*   We calculate a **Sentiment Score** from `-1.0` (Strongly Negative) to `+1.0` (Strongly Positive) using the formula: `Score = P(Positive) - P(Negative)`.

### Successful Test Cases
The model handles complex psychological tests, including:
*   **Direct Negative:** *"The product broke after one day."* → **Negative (-0.99)**
*   **Mixed Long:** *"The atmosphere is beautiful... But then the food arrived. My $40 steak was overcooked..."* → **Negative (-0.99)**
*   **Indirect/Sarcastic:** *"If you enjoy waiting 45 minutes for a cold burger and rude service, this is your paradise."* → **Negative**

### Frontend (Gradio)
The model is deployed on Hugging Face Spaces. The frontend features a large text box allowing users to paste anything from a 3-word sentence to a 300-word essay. The backend processes it and returns the label and score instantly.

---

## 📁 File Structure
*   `app.py`: The Gradio frontend code.
*   `requirements.txt`: Dependencies (`transformers`, `torch`, `gradio`).
*   `config.json`: Model configuration.
*   `model.safetensors`: The 90MB file containing the 22.7 Million fine-tuned weights.
*   `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`: Text processing files.
*   `README.md`: This documentation.

---
*Developed as a complete end-to-end NLP pipeline: Data Mixing → Text Preprocessing → Full Transformer Fine-Tuning → Hosting via Gradio.*