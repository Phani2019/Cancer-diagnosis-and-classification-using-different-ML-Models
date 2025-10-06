# Personalized Cancer Diagnosis and Classification

This project focuses on classifying genetic mutations based on clinical evidence from text-based literature. It utilizes various machine learning models to predict the class of a given genetic variation.

## Project Overview

- **Source:** [Kaggle: MSKCC Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment/)
- **Data Provider:** Memorial Sloan Kettering Cancer Center (MSKCC)
- **Context:** The goal is to classify genetic variations/mutations using text-based clinical literature, which is a crucial step in personalized cancer treatment.

## Problem Statement

Classify given genetic variations/mutations into one of nine classes based on evidence from text-based clinical literature.

## Data

The dataset consists of two files:
- `training_variants`: Contains information about genetic mutations.
- `training_text`: Contains the clinical evidence (text) used by experts to classify the mutations.

Both files share a common `ID` column, which links the genetic variation to its corresponding clinical text.

## Machine Learning Problem Formulation

- **Type of ML Problem:** Multi-class classification (nine classes).
- **Performance Metric:**
  - Multi-class log-loss
  - Confusion matrix
- **Objectives & Constraints:**
  - **Objective:** Predict the probability of each data point belonging to each of the nine classes.
  - **Constraints:**
    - Interpretability is important.
    - Class probabilities are required.
    - Log-loss is used to penalize errors in class probabilities.
    - No strict latency constraints.

## Dataset Split

The dataset is split into three parts:
- **Training set:** 64%
- **Cross-validation set:** 16%
- **Test set:** 20%

## Exploratory Data Analysis (EDA)

The EDA process included:
- Reading and merging the gene, variation, and text data.
- Preprocessing the text data.
- Analyzing the distribution of classes in the train, test, and cross-validation sets.
- Performing univariate analysis on the 'Gene' feature.

## Machine Learning Models

Several machine learning models were trained and evaluated to find the best approach for this classification problem.

### 1. Baseline Model: Naive Bayes
- A baseline model using Naive Bayes was implemented, as it is well-suited for text data.

### 2. K-Nearest Neighbour (KNN)
- A KNN model was tested, using response coding to handle high-dimensional data.

### 3. Logistic Regression
- Logistic Regression was implemented with and without class balancing to handle the imbalanced dataset.

### 4. Linear Support Vector Machines (SVM)
- A Linear SVM with class balancing was used due to its interpretability and performance with high-dimensional data.

### 5. Random Forest
- A Random Forest classifier was tested with both one-hot encoding and response coding.

### 6. Ensemble Models
- **Stacking:** A stacked ensemble of Logistic Regression, SVM, and Naive Bayes was created with Logistic Regression as the meta-classifier.
- **Voting Classifier:** A maximum voting ensemble was also tested.

## Results

The performance of the different models was compared based on log-loss and the number of misclassified points. The following table summarizes the results on the test set:

| Model                                             | Log-Loss | Misclassified Points (%) |
|---------------------------------------------------|----------|--------------------------|
| Random Model                                      | -        | -                        |
| Naive Bayes                                       | 1.253    | 42.6%                    |
| K-Nearest Neighbour                               | 1.037    | 36.5%                    |
| Logistic Regression (with class balancing)        | 1.094    | 35.2%                    |
| Logistic Regression (without class balancing)     | 1.063    | 35.3%                    |
| Linear SVM (with class balancing)                 | 1.116    | 37.0%                    |
| Random Forest (One-hot encoding)                  | 1.181    | 41.4%                    |
| Random Forest (Response coding)                   | 1.335    | 49.2%                    |
| Stacking Classifier                               | 1.145    | 38.6%                    |
| Maximum Voting Classifier                         | 1.206    | 38.5%                    |
| Logistic Regression (Unigrams & Bigrams)          | 1.103    | 36.3%                    |
| **Logistic Regression (with all features)**         | **0.998**| **32.3%**                |

**Conclusion:** The **Logistic Regression model with all features (Gene, Variation, and Text)** performed the best, achieving the lowest log-loss and the fewest misclassified points. This model is also interpretable, which satisfies one of the key project constraints.