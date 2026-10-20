---
license: apache-2.0
task_categories:
- text-classification
tags:
- sentiment analysis
- reviews
- text data
- nlp
- yelp
- binary
pretty_name: Yelp Reviews for Binary Senti Analysis
language:
- en
size_categories:
- 1M<n<10M
---
# Dataset Card for Dataset Name

The Yelp reviews polarity dataset is constructed by considering stars 1 and 2 negative, and 3 and 4 positive. For each polarity 280,000 training samples and 19,000 testing samples are taken randomly. In total there are 560,000 training samples and 38,000 testing samples. Negative polarity is class 1, and positive class 2.

### Dataset Description

The files train.csv and test.csv contain all the training samples as comma-separated values. There are 2 columns in them, corresponding to class index (1 and 2) and review text. The review texts are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".
- **License:** Apache 2

### Dataset Sources

- **Link on Kaggle:** https://www.kaggle.com/datasets/yacharki/yelp-reviews-for-sentianalysis-binary-np-csv
- **DOI:** @misc{xiang_zhang_yassir_acharki_2022,
	title={🏪Yelp Reviews for Senti-Analysis Binary -N/P+},
	url={https://www.kaggle.com/dsv/3603961},
	DOI={10.34740/KAGGLE/DSV/3603961},
	publisher={Kaggle},
	author={Xiang Zhang and Yassir Acharki},
	year={2022}
}
## Uses

NLP

### Direct Use

Binary sentiment analysis


## Dataset Structure

The Dataset Contains 

Readme.md

test.csv

train.csv


## Dataset Card Contact

For more info visit : 

https://www.kaggle.com/datasets/yacharki/yelp-reviews-for-sentianalysis-binary-np-csv