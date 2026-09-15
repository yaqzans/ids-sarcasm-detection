# IDS Final Term Project — Sarcasm Detection (Project Idea 5)

Binary sarcasm detection on English tweets, comparing Bag of Words against TF-IDF
across four traditional classifiers.

## Folder layout

| Folder | What is in it |
|---|---|
| `run/` | **everything needed to run the project.** Nothing outside this folder is required |
| `submission/` | the four files to hand in |
| `report/` | LaTeX source, figures and build scripts for the paper |
| `archive/` | development probes and logs, kept for the record, not needed to run |
| `materials/` | the assignment brief, the course slides, and other groups' projects |

## To run it

Everything lives in `run/`. Open a terminal there and run:

```bash
"D:/Apps/R/R-4.6.0/bin/Rscript.exe" ids_final_project_group_XX.r > run_output.txt 2>&1
```

R 4.6.0 is at `D:\Apps\R\R-4.6.0` and is **not** on PATH, which is why the full path
is used. From RStudio, just open the script and source it.

Takes about two minutes. Line 1 is `setwd()`; change it if the folder moves.

**Inputs** (already in `run/`): `train.csv`, `test.csv`
**Outputs** (written into `run/`): `model_comparison_results.csv`,
`ids_final_dataset_sample_group_XX.csv`, `run_output.txt`

Packages needed, all already installed: `tm`, `SnowballC`, `naivebayes`, `glmnet`,
`LiblineaR`, `rpart`. The script does not install anything.

## Submission files

Replace `XX` with the real group number in all four names before submitting.

| File | What it is |
|---|---|
| `ids_final_project_group_XX.r` | the R program, 325 lines, no comments |
| `ids_final_project_group_XX.pdf` | the paper, IEEE format, 6 pages |
| `ids_final_dataset_sample_group_XX.csv` | 1,000-row representative sample |
| `ids_final_project_group_XX.zip` | the three files above |

## Results

Balanced sample of 30,000 tweets, 3,726 features, 23,843 train / 5,963 test.
Majority-class baseline is 0.5028.

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

Bag of Words beats TF-IDF for all three linear models.

## Rebuilding the paper

In `report/`:

```bash
python assemble.py
```

That stitches `preamble.tex`, `body.tex` and `floats.tex` into
`ids_final_project_group_XX.tex`. Then run `pdflatex`, `bibtex`, `pdflatex`,
`pdflatex` using `C:\Users\shamv\AppData\Roaming\TinyTeX\bin\windows\`.

Edit `body.tex`, never the assembled `.tex`, since assembling overwrites it.
`make_figures.py`, `make_flow.py` and `make_combined.py` regenerate the figures.

## Notes for the viva

- **Total dataset**: 89,536 tweets in 4 classes; 43,210 after reducing to
  sarcasm/regular and removing duplicates; 30,000 balanced sample modelled.
- **Total features**: 18,584 distinct training terms, 3,726 after removing very
  sparse terms.
- **Algorithms**: Multinomial Naive Bayes, Logistic Regression (glmnet,
  lambda = 0.002), linear SVM (LIBLINEAR, cost 0.01), Decision Tree (rpart).
- **Why the hashtags are removed**: 99.98% of sarcasm tweets carry a sarcasm
  hashtag and 99.99% of regular tweets carry a topic hashtag. The labels are
  almost perfectly recoverable from the hashtags alone, in both directions.
  Leaving them in would give above 99% accuracy that measures nothing.
- **Why stopwords are kept**: removing them costs 0.64 accuracy points, because
  negation words (not, no, never, but, just) are sarcasm cues. Step 7 of the
  program measures this directly.
- **Why the Decision Tree is worse**: it splits on one word at a time, and no
  single word carries enough evidence in a 3,726-dimensional sparse space. The
  linear models sum weak evidence across all terms at once.
- **Why BoW beats TF-IDF**: idf down-weights frequent words, but sarcasm cues
  (love, great, just, yeah) are frequent by construction. Tweets are also short,
  so the tf denominator is small and unstable.
- **Live demo**: `classify_tweet("your sentence here")` at the R console after the
  script finishes. Step 10 runs four examples and gets all four right.
