# EMAIL_Spam-Not_Spam — Email Spam Classification

Project summary
This repository contains an end-to-end email spam classification example implemented as Jupyter Notebook(s). The project demonstrates the supervised-learning workflow for text classification: data inspection, text preprocessing (cleaning, tokenization, vectorization), train/test split, model training and evaluation (baseline models such as Naive Bayes / Logistic Regression), and single-sample inference. It is intended as a concise, reproducible notebook-based project for learning and quick experimentation.

What I learned
- How to preprocess raw email text: cleaning, lowercasing, tokenization, stop-word removal, and optional lemmatization/stemming.
- How to convert text to numeric features using CountVectorizer and TF-IDF for supervised learning.
- How to split data with stratification to preserve label balance and ensure fair evaluation.
- Training and comparing baseline classifiers for text data (e.g., Multinomial Naive Bayes, Logistic Regression).
- Evaluating models using accuracy, precision, recall, F1-score, and confusion matrices to account for imbalanced classes.
- How to perform inference on a custom email message by applying the same preprocessing and vectorization pipeline.
- Importance of reproducibility (fixed random seed) and documenting preprocessing steps used in the model pipeline.

What I used
- Python (pandas, numpy)
- scikit-learn (train_test_split, CountVectorizer / TfidfVectorizer, MultinomialNB, LogisticRegression, metrics)
- (Optional) nltk or spaCy for tokenization/stopwords/lemmatization
- Jupyter Notebook(s) containing the full analysis and code
- Dataset: a CSV or TSV of emails with labels (spam / not spam)

Suggested project structure
- notebooks/ — exploratory notebooks and the main training notebook
- data/ — raw and cleaned dataset files (not committed if large or private)
- scripts/ — optional scripts extracted from the notebook (train.py, predict.py)
- requirements.txt — dependencies for reproducibility
- README.md — this file

Notes on the dataset
- Typical format: one row per email, a text column containing the message, and a label column with values like "spam"/"ham" or 1/0.
- Inspect for class imbalance, missing text fields, and duplicated messages before modeling.
- Small or imbalanced datasets require careful evaluation (use precision/recall and possibly cross-validation).

Methodology / Workflow
1. Load data
   - Read the dataset into a pandas DataFrame and inspect (head, shape, isnull, value_counts).
2. Preprocess text
   - Clean text (remove headers/HTML if present), lowercase, remove punctuation and stop words; optionally apply stemming or lemmatization.
3. Feature extraction
   - Convert text to numeric vectors using CountVectorizer or TfidfVectorizer; consider n-grams and max_features tuning.
4. Train / test split
   - Use train_test_split(test_size=0.2, stratify=y, random_state=SEED) to preserve label proportions.
5. Model training
   - Train baseline models (MultinomialNB, LogisticRegression); compare performance.
6. Evaluation
   - Compute accuracy, precision, recall, F1-score, and view the confusion matrix. Consider ROC/AUC for threshold analysis.
7. Inference
   - Apply the same preprocessing + vectorizer to new messages and run model.predict to classify spam vs. not-spam.


