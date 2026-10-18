# sentiment-analysis
This repository includes the dataset of mobile app(Facebook, WhatsApp and Instagram) reviews.
# IGWA Dataset: Instagram and WhatsApp Google Play Reviews

This repository contains the **IGWA dataset (Instagram and WhatsApp Reviews)** used in the paper:

> *Enhancing Sentiment Analysis of Mobile App Reviews: Handling Emojis and Slang with BERT*  
> Accepted at DS4EIW 2026 – IEEE ICDE Workshop on Data Science for Equality, Inclusion and Well-being.

---

## 📌 Overview

The IGWA dataset consists of user reviews collected from the **Google Play Store** for two widely used mobile applications:

- Instagram  
- WhatsApp  

The dataset was constructed to evaluate the impact of **emoji normalization** and **slang normalization** on BERT-based sentiment analysis.

The reviews are labeled into three sentiment classes:

- Negative  
- Neutral  
- Positive  

Labels are derived from star ratings following standard sentiment mapping practices.

---

## 📥 Data Collection Methodology

The dataset was collected using the **google-play-scraper** Python library, which programmatically retrieves publicly available review content from the Google Play Store.

Retrieved fields included:

- Review text  
- Star rating  
- Timestamp (later anonymized)  
- Application identifier  

Only publicly accessible reviews were collected. No private or restricted data were accessed.

---

## 📊 Dataset Files

This repository contains:

- `dataset_IGWA_only.csv`  
  → Instagram and WhatsApp reviews used in the main experiments.

- `dataset_IGWA_FB10.csv`  
  → Extended dataset including additional reviews for comparative evaluation.

Each CSV file includes the following fields (after preprocessing):

- `original_text`  
- `rating`  
- `sentiment_text` (Negative / Neutral / Positive)
- 'sentiment_id' (0/1/2)
- 'rating'
- lang_detect(en)
- `app_name`  
- `has_emoji` (binary indicator)  
- `has_slang` (binary indicator)  

All personally identifiable information (e.g., usernames, user IDs, profile links) has been removed.

---

## 🧹 Preprocessing

The following preprocessing steps were applied:

1. Removal of URLs and non-text artifacts  
2. Emoji normalization (conversion to semantic textual descriptions)  
3. Slang normalization (dictionary-based expansion of informal expressions)  
4. Tokenization using BERT tokenizer  
5. Label assignment based on star ratings  

These preprocessing strategies are described in detail in the associated paper.

---

## 🔎 Dataset Statistics
- Total number of reviews: 54,544
- Train / Val / Test :  43,634 / 5,455 / 5,455 
- Negative: 20,767 (38.07%)
- Neutral: 3,646 (6.68%)
- Positive: 30,131 (55.24%) 
- Reviews containing emojis: 11.25\% 
- Reviews containing slang: 0.29\%  
- Average review length: 15.44 tokens  

---

## ⚖ Ethical Considerations

- The dataset consists exclusively of publicly available reviews.
- All personally identifiable information has been removed.
- The dataset is shared strictly for **academic and research purposes**.
- Redistribution for commercial purposes is not permitted.

Researchers using this dataset are responsible for complying with Google Play’s Terms of Service.

---

## 🔁 Reproducibility

To reproduce the dataset:

1. Install `google-play-scraper` in Python.
2. Use the provided scraping methodology described in the paper.
3. Apply the preprocessing steps outlined above.

Future updates may include the scraping script and preprocessing pipeline.

---

## 📖 Citation



