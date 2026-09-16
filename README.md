# Sarcasm Detection in Social Media Text

Binary sarcasm detection on English tweets in R, comparing **Bag of Words** against
**TF-IDF** across four traditional classifiers.

Coursework for Introduction to Data Science, Summer 2025-26.

## What is in this repository

Only what is needed to run the program.

```
run/
  ids_final_project_group_XX.r   the whole program, 325 lines
  train.csv                      dataset, 81,408 tweets
  test.csv                       dataset, 8,128 tweets
```

The paper, the viva notes and the submission bundle are course deliverables and are
deliberately not published here.

## To run it

```bash
cd run
Rscript ids_final_project_group_XX.r > run_output.txt 2>&1
```

Line 1 of the script is `setwd()`. Change it if the folder moves.

Needs six packages: `tm`, `SnowballC`, `naivebayes`, `glmnet`, `LiblineaR`, `rpart`.
Install with:

```r
install.packages(c("tm","SnowballC","naivebayes","glmnet","LiblineaR","rpart"))
```

Takes about two minutes on a laptop CPU. No GPU needed.

Writes three files into `run/`: `model_comparison_results.csv`,
`ids_final_dataset_sample_group_XX.csv` and `run_output.txt`.

## The dataset

Kaggle, ["Tweets with Sarcasm and Irony"](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony),
which traces back to Ling and Klinger (ESWC 2016). 89,536 tweets in four classes:
figurative, irony, sarcasm and regular. Only sarcasm and regular are used here.

**The labels leak through the collection hashtags.** The corpus was gathered by
searching for marker hashtags, so 99.98% of sarcastic tweets still contain a sarcasm
hashtag and 99.99% of regular tweets contain one of the eight topic hashtags used to
collect them. Leaving those in gives above 99% accuracy that measures nothing. Every
hashtag is stripped before training, and the accuracy below is what survives that.

## What the program does

1. Loads both CSVs, explores the data, and measures the hashtag leakage
2. Reduces to sarcasm and regular, removes duplicates, samples 15,000 of each
3. Cleans the text: lowercase, strip URLs, mentions, hashtags, punctuation, digits,
   then Porter stemming
4. Splits 80/20 **before** building any vocabulary
5. Builds Bag of Words from the training tweets only, then TF-IDF from it
6. Trains four classifiers on each representation, eight combinations
7. Reports a confusion matrix and accuracy, precision, recall and F1 for each
8. Runs a stopword-removal ablation and a live demo on unseen sentences

## Results

30,000 tweets, 3,726 features, 23,843 train / 5,963 test. Majority baseline 0.5028.

| Representation | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| BoW | Naive Bayes | 0.7696 | 0.7392 | 0.8369 | **0.7850** |
| BoW | Logistic Regression | **0.7775** | 0.7916 | 0.7565 | 0.7737 |
| BoW | SVM | 0.7744 | 0.7938 | 0.7448 | 0.7685 |
| BoW | Decision Tree | 0.6530 | 0.7294 | 0.4927 | 0.5881 |
| TF-IDF | Naive Bayes | 0.7582 | 0.7231 | 0.8412 | 0.7777 |
| TF-IDF | Logistic Regression | 0.7696 | 0.7633 | 0.7852 | 0.7741 |
| TF-IDF | SVM | 0.7672 | 0.7480 | 0.8099 | 0.7777 |
| TF-IDF | Decision Tree | 0.6646 | 0.7127 | 0.5577 | 0.6257 |

Bag of Words beats TF-IDF for all three linear models, by 0.72 to 1.14 points. The
decision tree trails by about 11 points because it splits on one word at a time, and
no single word carries enough evidence in a sparse 3,726-dimensional space.

Two controlled findings:

- **Keeping stopwords is worth +0.64 accuracy points.** Stopword lists delete *not*,
  *no*, *never*, *but* and *just*, which are sarcasm cues. TF-IDF already discounts
  common words by construction.
- 0.7775 sits close to the 75.4% that Bamman and Smith (2015) reported from tweet text
  alone after they also removed the hashtags. Higher figures in the literature come
  from setups that kept the markers.

## Live demo

After the script finishes, at the R console:

```r
classify_tweet("Oh great, another Monday morning meeting. Just what I always wanted.")
# "sarcasm"
```
