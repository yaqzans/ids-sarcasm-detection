setwd("D:/AIUB/10. Summer 2026/Final/Data Science/Project/run")

library(tm)
library(SnowballC)
library(naivebayes)
library(glmnet)
library(LiblineaR)
library(rpart)

set.seed(123)

cat("\n===== STEP 1: DATA COLLECTION AND EXPLORATION =====\n\n")

train_raw <- read.csv("train.csv", stringsAsFactors = FALSE)
test_raw <- read.csv("test.csv", stringsAsFactors = FALSE)
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

sample_size <- 15000
sarcasm_rows <- which(df$class == "sarcasm")
regular_rows <- which(df$class == "regular")
picked <- c(sample(sarcasm_rows, sample_size), sample(regular_rows, sample_size))
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
  corpus <- tm_map(corpus, stemDocument)
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}

corpus <- clean_corpus(VCorpus(VectorSource(df$tweets)))

cat("Example tweets after preprocessing:\n")
print(head(sapply(corpus, as.character), 3))

cat("\n===== STEP 3: TRAIN AND TEST SPLIT =====\n\n")

label <- factor(df$class, levels = c("regular", "sarcasm"))

n <- length(corpus)
train_index <- sample(1:n, round(0.8 * n))

train_corpus <- corpus[train_index]
test_corpus <- corpus[-train_index]
y_train <- label[train_index]
y_test <- label[-train_index]

cat("Training tweets:\n")
print(length(train_corpus))
cat("Testing tweets:\n")
print(length(test_corpus))

cat("\n===== STEP 4: BAG OF WORDS AND TF-IDF =====\n\n")

dtm_train <- DocumentTermMatrix(train_corpus)
cat("Distinct terms in the training tweets:\n")
print(ncol(dtm_train))

dtm_train <- removeSparseTerms(dtm_train, 0.9998)
vocab <- Terms(dtm_train)
cat("\nVocabulary size after removing very sparse terms:\n")
print(length(vocab))

bow_train <- as.matrix(dtm_train)
bow_test <- as.matrix(DocumentTermMatrix(test_corpus, control = list(dictionary = vocab)))

keep_train <- rowSums(bow_train) > 0
keep_test <- rowSums(bow_test) > 0
cat("\nTweets dropped for containing no vocabulary term:\n")
print(sum(!keep_train) + sum(!keep_test))

bow_train <- bow_train[keep_train, ]
bow_test <- bow_test[keep_test, ]
y_train <- y_train[keep_train]
y_test <- y_test[keep_test]

idf <- log(nrow(bow_train) / colSums(bow_train > 0))

tfidf_train <- sweep(bow_train / rowSums(bow_train), 2, idf, "*")
tfidf_test <- sweep(bow_test / rowSums(bow_test), 2, idf, "*")

cat("\nBoW training matrix size:\n")
print(dim(bow_train))
cat("BoW testing matrix size:\n")
print(dim(bow_test))

cat("\nFive highest IDF weights (rarest training terms):\n")
print(round(sort(idf, decreasing = TRUE)[1:5], 4))
cat("\nFive lowest IDF weights (commonest training terms):\n")
print(round(sort(idf)[1:5], 4))

cat("\nFirst training tweet in both representations, five strongest terms:\n")
print(sort(bow_train[1, ], decreasing = TRUE)[1:5])
print(round(sort(tfidf_train[1, ], decreasing = TRUE)[1:5], 4))

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
lr_bow <- glmnet(bow_train, y_train, family = "binomial", lambda = 0.002)
pred_lr_bow <- factor(predict(lr_bow, bow_test, type = "class"),
                      levels = c("regular", "sarcasm"))
s2 <- get_scores(y_test, pred_lr_bow)

cat("----- BoW + SVM -----\n")
svm_bow <- LiblineaR(bow_train, y_train, type = 2, cost = 0.01)
pred_svm_bow <- factor(predict(svm_bow, bow_test)$predictions,
                       levels = c("regular", "sarcasm"))
s3 <- get_scores(y_test, pred_svm_bow)

cat("----- TF-IDF + Naive Bayes -----\n")
nb_tfidf <- multinomial_naive_bayes(tfidf_train, y_train, laplace = 1)
pred_nb_tfidf <- predict(nb_tfidf, tfidf_test)
s4 <- get_scores(y_test, pred_nb_tfidf)

cat("----- TF-IDF + Logistic Regression -----\n")
lr_tfidf <- glmnet(tfidf_train, y_train, family = "binomial", lambda = 0.002)
pred_lr_tfidf <- factor(predict(lr_tfidf, tfidf_test, type = "class"),
                        levels = c("regular", "sarcasm"))
s5 <- get_scores(y_test, pred_lr_tfidf)

cat("----- TF-IDF + SVM -----\n")
svm_tfidf <- LiblineaR(tfidf_train, y_train, type = 2, cost = 0.01)
pred_svm_tfidf <- factor(predict(svm_tfidf, tfidf_test)$predictions,
                         levels = c("regular", "sarcasm"))
s6 <- get_scores(y_test, pred_svm_tfidf)

bow_train_df <- as.data.frame(bow_train)
bow_test_df <- as.data.frame(bow_test)
names(bow_test_df) <- names(bow_train_df)
bow_train_df$label <- y_train

tfidf_train_df <- as.data.frame(tfidf_train)
tfidf_test_df <- as.data.frame(tfidf_test)
names(tfidf_test_df) <- names(tfidf_train_df)
tfidf_train_df$label <- y_train

cat("----- BoW + Decision Tree -----\n")
tree_bow <- rpart(label ~ ., data = bow_train_df, method = "class",
                  control = rpart.control(xval = 0))
pred_tree_bow <- predict(tree_bow, bow_test_df, type = "class")
s7 <- get_scores(y_test, pred_tree_bow)

cat("----- TF-IDF + Decision Tree -----\n")
tree_tfidf <- rpart(label ~ ., data = tfidf_train_df, method = "class",
                    control = rpart.control(xval = 0))
pred_tree_tfidf <- predict(tree_tfidf, tfidf_test_df, type = "class")
s8 <- get_scores(y_test, pred_tree_tfidf)

rm(bow_train_df, bow_test_df, tfidf_train_df, tfidf_test_df)
invisible(gc())

cat("\n===== STEP 6: COMPARISON OF REPRESENTATIONS AND MODELS =====\n\n")

results <- data.frame(
  Representation = c("BoW", "BoW", "BoW", "BoW",
                     "TF-IDF", "TF-IDF", "TF-IDF", "TF-IDF"),
  Model = c("Naive Bayes", "Logistic Regression", "SVM", "Decision Tree",
            "Naive Bayes", "Logistic Regression", "SVM", "Decision Tree"),
  Accuracy = round(c(s1[1], s2[1], s3[1], s7[1], s4[1], s5[1], s6[1], s8[1]), 4),
  Precision = round(c(s1[2], s2[2], s3[2], s7[2], s4[2], s5[2], s6[2], s8[2]), 4),
  Recall = round(c(s1[3], s2[3], s3[3], s7[3], s4[3], s5[3], s6[3], s8[3]), 4),
  F1 = round(c(s1[4], s2[4], s3[4], s7[4], s4[4], s5[4], s6[4], s8[4]), 4))

print(results, row.names = FALSE)

baseline <- max(table(y_test)) / length(y_test)
cat("\nMajority-class baseline accuracy:\n")
print(round(baseline, 4))

comparison <- data.frame(
  Model = c("Naive Bayes", "Logistic Regression", "SVM", "Decision Tree"),
  BoW = results$Accuracy[1:4],
  TF_IDF = results$Accuracy[5:8])
comparison$Difference <- round(comparison$BoW - comparison$TF_IDF, 4)

cat("\nBoW against TF-IDF accuracy for each classifier:\n")
print(comparison, row.names = FALSE)

best <- results[which.max(results$F1), ]
cat("\nBest combination by F1:\n")
print(best, row.names = FALSE)

cat("\n===== STEP 7: EFFECT OF STOPWORD REMOVAL =====\n\n")

stop_corpus <- tm_map(corpus, removeWords, stopwords("english"))
stop_corpus <- tm_map(stop_corpus, stripWhitespace)

stop_dtm_train <- removeSparseTerms(DocumentTermMatrix(stop_corpus[train_index]), 0.9998)
stop_vocab <- Terms(stop_dtm_train)
stop_bow_train <- as.matrix(stop_dtm_train)
stop_bow_test <- as.matrix(DocumentTermMatrix(stop_corpus[-train_index],
                                              control = list(dictionary = stop_vocab)))

stop_keep_train <- rowSums(stop_bow_train) > 0
stop_keep_test <- rowSums(stop_bow_test) > 0
stop_bow_train <- stop_bow_train[stop_keep_train, ]
stop_bow_test <- stop_bow_test[stop_keep_test, ]

stop_nb <- multinomial_naive_bayes(stop_bow_train, label[train_index][stop_keep_train], laplace = 1)
pred_stop_nb <- predict(stop_nb, stop_bow_test)
stop_scores <- get_scores(label[-train_index][stop_keep_test], pred_stop_nb)

cat("Vocabulary with stopwords removed:", length(stop_vocab), "\n")
cat("Vocabulary with stopwords retained:", length(vocab), "\n")
cat("Naive Bayes accuracy with stopwords removed:", round(stop_scores[1], 4), "\n")
cat("Naive Bayes accuracy with stopwords retained:", round(s1[1], 4), "\n")

cat("\n===== STEP 8: OUTPUT FILES =====\n\n")

write.csv(results, "model_comparison_results.csv", row.names = FALSE)

sample_index <- sample(1:nrow(df), 1000)
sample_df <- data.frame(tweet = df$tweets[sample_index],
                        class = df$class[sample_index],
                        stringsAsFactors = FALSE)
write.csv(sample_df, "ids_final_dataset_sample_group_XX.csv", row.names = FALSE)

cat("Sample dataset rows written:\n")
print(nrow(sample_df))
print(table(sample_df$class))

cat("\n===== STEP 9: PROJECT SUMMARY =====\n\n")

cat("Total tweets in the downloaded dataset:", total_tweets, "\n")
cat("Tweets after keeping sarcasm and regular and removing duplicates:", binary_tweets, "\n")
cat("Balanced sample drawn for modelling:", sample_size * 2, "\n")
cat("Tweets actually modelled:", nrow(bow_train) + nrow(bow_test), "\n")
cat("Training tweets:", length(y_train), "\n")
cat("Testing tweets:", length(y_test), "\n")
cat("Total features (vocabulary size):", length(vocab), "\n")
cat("Representations compared: Bag of Words and TF-IDF\n")
cat("Algorithms used: Multinomial Naive Bayes, Logistic Regression,",
    "Linear SVM, Decision Tree\n")
cat("Majority-class baseline accuracy:", round(baseline, 4), "\n")
cat("Best combination:", best$Representation, "+", best$Model,
    "with accuracy", best$Accuracy, "and F1", best$F1, "\n")

cat("\n===== STEP 10: LIVE DEMONSTRATION =====\n\n")

classify_tweet <- function(text) {
  new_corpus <- clean_corpus(VCorpus(VectorSource(text)))
  new_bow <- as.matrix(DocumentTermMatrix(new_corpus, control = list(dictionary = vocab)))
  prediction <- predict(svm_bow, new_bow)$predictions
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
