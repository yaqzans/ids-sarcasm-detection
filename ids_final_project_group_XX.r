setwd("D:/AIUB/10. Summer 2026/Final/Data Science/Project")

library(tm)
library(SnowballC)
library(naivebayes)
library(glmnet)
library(e1071)

set.seed(123)

cat("\n===== STEP 1: DATA COLLECTION AND EXPLORATION =====\n\n")

train_raw <- read.csv("data/train.csv", stringsAsFactors = FALSE)
test_raw <- read.csv("data/test.csv", stringsAsFactors = FALSE)
df <- rbind(train_raw, test_raw)
total_tweets <- nrow(df)

print(dim(df))
str(df)
print(table(df$class))
print(colSums(is.na(df)))

df$word_count <- lengths(strsplit(df$tweets, " "))
print(summary(df$word_count))

cat("\n----- Hashtag check: how the dataset was collected -----\n\n")

df$sarcasm_tag <- grepl("#sarcas|#irony|#ironic|#not", df$tweets, ignore.case = TRUE)
df$regular_tag <- grepl("#news|#politics|#education|#peace|#drugs|#humor|#late|#gopdebate",
                        df$tweets, ignore.case = TRUE)

cat("Proportion of each class carrying a sarcasm/irony hashtag:\n")
print(round(prop.table(table(df$class, df$sarcasm_tag), 1), 4))

cat("\nProportion of each class carrying a regular-topic hashtag:\n")
print(round(prop.table(table(df$class, df$regular_tag), 1), 4))

cat("\n----- Reduce to a binary sarcasm task -----\n\n")

df <- df[df$class == "sarcasm" | df$class == "regular", ]
df <- df[!duplicated(df$tweets), ]
binary_tweets <- nrow(df)
print(table(df$class))

sarcasm_rows <- which(df$class == "sarcasm")
regular_rows <- which(df$class == "regular")
picked <- c(sample(sarcasm_rows, 5000), sample(regular_rows, 5000))
df <- df[picked, ]

cat("\nBalanced sample used for modelling:\n")
print(table(df$class))

cat("\nExample raw tweets:\n")
print(head(df$tweets, 3))

cat("\n===== STEP 2: TEXT PREPROCESSING =====\n\n")

remove_pattern <- content_transformer(function(x, pattern) gsub(pattern, " ", x))

clean_corpus <- function(corpus) {
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, remove_pattern, "http[^ ]*")
  corpus <- tm_map(corpus, remove_pattern, "@[a-z0-9_]+")
  corpus <- tm_map(corpus, remove_pattern, "#[a-z0-9_]+")
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stemDocument)
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}

corpus <- clean_corpus(VCorpus(VectorSource(df$tweets)))

cat("Example tweets after preprocessing:\n")
print(head(sapply(corpus, as.character), 3))

cat("\n===== STEP 3: BAG OF WORDS AND TF-IDF =====\n\n")

dtm <- DocumentTermMatrix(corpus)
cat("Distinct terms before sparse-term removal:\n")
print(ncol(dtm))

dtm <- removeSparseTerms(dtm, 0.999)
vocab <- Terms(dtm)
cat("\nVocabulary size after sparse-term removal:\n")
print(length(vocab))

bow <- as.matrix(dtm)

keep <- rowSums(bow) > 0
cat("\nTweets dropped for containing no vocabulary term:\n")
print(sum(!keep))

bow <- bow[keep, ]
label <- factor(df$class[keep], levels = c("regular", "sarcasm"))

dtm_tfidf <- DocumentTermMatrix(corpus[keep],
                                control = list(dictionary = vocab, weighting = weightTfIdf))
tfidf <- as.matrix(dtm_tfidf)

cat("\nBoW matrix size:\n")
print(dim(bow))
cat("TF-IDF matrix size:\n")
print(dim(tfidf))

cat("\nSame tweet in both representations, five most weighted terms:\n")
print(sort(bow[1, ], decreasing = TRUE)[1:5])
print(round(sort(tfidf[1, ], decreasing = TRUE)[1:5], 4))

cat("\n===== STEP 4: TRAIN AND TEST SPLIT =====\n\n")

n <- nrow(bow)
train_index <- sample(1:n, round(0.8 * n))

bow_train <- bow[train_index, ]
bow_test <- bow[-train_index, ]
tfidf_train <- tfidf[train_index, ]
tfidf_test <- tfidf[-train_index, ]
y_train <- label[train_index]
y_test <- label[-train_index]

cat("Training tweets:\n")
print(length(y_train))
cat("Testing tweets:\n")
print(length(y_test))
print(table(Training = y_train))
print(table(Testing = y_test))

cat("\n===== STEP 5: CLASSIFICATION AND EVALUATION =====\n\n")

get_scores <- function(actual, predicted) {
  cm <- table(Actual = actual, Predicted = predicted)
  print(cm)
  tp <- cm["sarcasm", "sarcasm"]
  tn <- cm["regular", "regular"]
  fp <- cm["regular", "sarcasm"]
  fn <- cm["sarcasm", "regular"]
  accuracy <- (tp + tn) / sum(cm)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1 <- 2 * precision * recall / (precision + recall)
  scores <- c(Accuracy = accuracy, Precision = precision, Recall = recall, F1 = f1)
  cat("\n")
  print(round(scores, 4))
  cat("\n")
  return(scores)
}

cat("----- BoW + Naive Bayes -----\n")
nb_bow <- multinomial_naive_bayes(bow_train, y_train, laplace = 1)
pred_nb_bow <- predict(nb_bow, bow_test)
s1 <- get_scores(y_test, pred_nb_bow)

cat("----- BoW + Logistic Regression -----\n")
lr_bow <- cv.glmnet(bow_train, y_train, family = "binomial", nfolds = 5)
pred_lr_bow <- factor(predict(lr_bow, bow_test, s = "lambda.min", type = "class"),
                      levels = c("regular", "sarcasm"))
s2 <- get_scores(y_test, pred_lr_bow)

cat("----- BoW + SVM -----\n")
svm_bow <- svm(bow_train, y_train, kernel = "linear", scale = FALSE)
pred_svm_bow <- predict(svm_bow, bow_test)
s3 <- get_scores(y_test, pred_svm_bow)

cat("----- TF-IDF + Naive Bayes -----\n")
nb_tfidf <- multinomial_naive_bayes(tfidf_train, y_train, laplace = 1)
pred_nb_tfidf <- predict(nb_tfidf, tfidf_test)
s4 <- get_scores(y_test, pred_nb_tfidf)

cat("----- TF-IDF + Logistic Regression -----\n")
lr_tfidf <- cv.glmnet(tfidf_train, y_train, family = "binomial", nfolds = 5)
pred_lr_tfidf <- factor(predict(lr_tfidf, tfidf_test, s = "lambda.min", type = "class"),
                        levels = c("regular", "sarcasm"))
s5 <- get_scores(y_test, pred_lr_tfidf)

cat("----- TF-IDF + SVM -----\n")
svm_tfidf <- svm(tfidf_train, y_train, kernel = "linear", scale = FALSE)
pred_svm_tfidf <- predict(svm_tfidf, tfidf_test)
s6 <- get_scores(y_test, pred_svm_tfidf)

cat("\n===== STEP 6: COMPARISON OF REPRESENTATIONS AND MODELS =====\n\n")

results <- data.frame(
  Representation = c("BoW", "BoW", "BoW", "TF-IDF", "TF-IDF", "TF-IDF"),
  Model = c("Naive Bayes", "Logistic Regression", "SVM",
            "Naive Bayes", "Logistic Regression", "SVM"),
  Accuracy = round(c(s1[1], s2[1], s3[1], s4[1], s5[1], s6[1]), 4),
  Precision = round(c(s1[2], s2[2], s3[2], s4[2], s5[2], s6[2]), 4),
  Recall = round(c(s1[3], s2[3], s3[3], s4[3], s5[3], s6[3]), 4),
  F1 = round(c(s1[4], s2[4], s3[4], s4[4], s5[4], s6[4]), 4))

print(results, row.names = FALSE)

baseline <- max(table(y_test)) / length(y_test)
cat("\nMajority-class baseline accuracy:\n")
print(round(baseline, 4))

comparison <- data.frame(
  Model = c("Naive Bayes", "Logistic Regression", "SVM"),
  BoW = results$Accuracy[1:3],
  TF_IDF = results$Accuracy[4:6])
comparison$Difference <- round(comparison$BoW - comparison$TF_IDF, 4)

cat("\nBoW against TF-IDF accuracy for each classifier:\n")
print(comparison, row.names = FALSE)

best <- results[which.max(results$F1), ]
cat("\nBest combination by F1:\n")
print(best, row.names = FALSE)

cat("\n===== STEP 7: OUTPUT FILES =====\n\n")

write.csv(results, "model_comparison_results.csv", row.names = FALSE)

sample_index <- sample(1:nrow(df), 1000)
sample_df <- data.frame(tweet = df$tweets[sample_index],
                        class = df$class[sample_index],
                        stringsAsFactors = FALSE)
write.csv(sample_df, "ids_final_dataset_sample_group_XX.csv", row.names = FALSE)

cat("Sample dataset rows written:\n")
print(nrow(sample_df))
print(table(sample_df$class))

cat("\n===== STEP 8: PROJECT SUMMARY =====\n\n")

cat("Total tweets in the downloaded dataset:", total_tweets, "\n")
cat("Tweets after keeping sarcasm and regular and removing duplicates:", binary_tweets, "\n")
cat("Balanced sample drawn for modelling:", 10000, "\n")
cat("Tweets actually modelled after preprocessing:", n, "\n")
cat("Training tweets:", length(y_train), "\n")
cat("Testing tweets:", length(y_test), "\n")
cat("Total features (vocabulary size):", length(vocab), "\n")
cat("Representations compared: Bag of Words and TF-IDF\n")
cat("Algorithms used: Multinomial Naive Bayes, Logistic Regression, Linear SVM\n")
cat("Majority-class baseline accuracy:", round(baseline, 4), "\n")
cat("Best combination:", best$Representation, "+", best$Model,
    "with accuracy", best$Accuracy, "and F1", best$F1, "\n")

cat("\n===== STEP 9: LIVE DEMONSTRATION =====\n\n")

classify_tweet <- function(text) {
  new_corpus <- clean_corpus(VCorpus(VectorSource(text)))
  new_bow <- as.matrix(DocumentTermMatrix(new_corpus, control = list(dictionary = vocab)))
  prediction <- predict(svm_bow, new_bow)
  return(as.character(prediction))
}

demo_tweets <- c(
  "Oh great, another Monday morning meeting. Just what I always wanted.",
  "The new library opens downtown this Saturday at ten in the morning.",
  "I love waiting two hours for a train that never shows up. Best day ever.",
  "Congratulations to the team on winning the championship last night.")

for (i in 1:length(demo_tweets)) {
  cat("Tweet:", demo_tweets[i], "\n")
  cat("Prediction:", classify_tweet(demo_tweets[i]), "\n\n")
}
