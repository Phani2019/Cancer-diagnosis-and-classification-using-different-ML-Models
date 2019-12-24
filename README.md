# Cancer-diagnosis-and-classification-using-different-ML-Models


Personalized cancer diagnosis
Business Problem
Description
Source: https://www.kaggle.com/c/msk-redefining-cancer-treatment/

Data: Memorial Sloan Kettering Cancer Center (MSKCC)

Download training_variants.zip and training_text.zip from Kaggle.

Context:
Source: https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/35336#198462

Problem statement :
Classify the given genetic variations/mutations based on evidence from text-based clinical literature.

Machine Learning Problem Formulation
Data
Data Overview
Source: https://www.kaggle.com/c/msk-redefining-cancer-treatment/data
We have two data files: one conatins the information about the genetic mutations and the other contains the clinical evidence (text) that human experts/pathologists use to classify the genetic mutations.
Both these data files are have a common column called ID
Mapping the real-world problem to an ML problem
Type of Machine Learning Problem
There are nine different classes a genetic mutation can be classified into => Multi class classification problem

Performance Metric
Source: https://www.kaggle.com/c/msk-redefining-cancer-treatment#evaluation

# Metric(s):

Multi class log-loss
Confusion matrix
Machine Learing Objectives and Constraints
Objective: Predict the probability of each data-point belonging to each of the nine classes.

# Constraints:

Interpretability
Class probabilities are needed.
Penalize the errors in class probabilites => Metric is Log-loss.
No Latency constraints.
Train, CV and Test Datasets
Split the dataset randomly into three parts train, cross validation and test with 64%,16%, 20% of data respectively

# Exploratory Data Analysis
Reading Data
Reading Gene and Variation Data
Reading Text Data
Preprocessing of text
Test, Train and Cross Validation Split
Splitting data into train, test and cross validation (64:20:16)
Distribution of y_i's in Train, Test and Cross Validation datasets
Prediction using a 'Random' Model
In a 'Random' Model, we generate the NINE class probabilites randomly such that they sum to 1.

Univariate Analysis
Univariate Analysis on Gene Feature
# Machine Learning Models
Base Line Model
# Naive Bayes
We know that for text data NB model is a baseline model. Now we apply the training data to the model and used the CV data for finding best hyper-parameter(alpha)

With the best alpha we fit the model. The test data is then applied to the model and we found out that the log-loss value is 1.25 which is quite less than Random Model. Here we also find out that the total number of mis-classified cases is 0.428. We also checked the probabilities of each class for each data and interpreted each point. This is to check why it is predicting particular class randomly. We conclude that for mis-classified points, the probability that the point belongs to a predicted class is very low. From the precision and recall matrix it is found out that most of points from class 2 predicted as 7. Similarly most points from class 1 are predicted as 4.


Log Loss : 1.253 Number of missclassified point : 0.425864661654133

# K Nearest Neighbour Classification


As we know that the k-NN model is not interpretable(which is our business constraint) but we still use this model just to find out the log loss values. Since k-NN suffers from the curse of dimensionality, we use response coding instead of one-hot encoding. After applying the data to the model we obtain the best hyper-parameter(k)


With the best k we fit the model and test data is applied to the model. The the log-loss value is 1.037 which is less than NB model. But number of mis-classified points are 36.47 percent(less than to NB model). In k-nn model it is found out that most of points from class 2 predicted as 7. Similarly most of points from class 1 predicted as 4.


Log loss : 1.0372973672339005 Number of mis-classified points : 0.36466165413533835

# Logistic Regression With Class balancing

As we have already seen, the LR model worked very well with univariate analysis. So we did some thorough analysis on LR by taking both imbalanced data and balanced data.

# With Class Balancing:
We also know that LR works well with high dimension data and it is also interpretable. So we did oversampling of lower class points and applied the training data to the model and used the CV data for finding best hyper-parameter (lambda)

With the best lambda we fitted the model and test data is applied to the model. The log-loss value is 1.048(close to k-nn). But number of mis-classified points are 34.77 percent(which are less than NB and K-nn). As LR is interpretable and mis-classified points are less than other models(k-NN and NB)it is better than k-NN and NB.
Without class balancing log loss and mis-classified points are increased. Therefore, we use class balancing.

Log loss : 1.0944298703192492 Number of mis-classified points : 0.35150375939849626

# Without Class balancing

Log loss : 1.062988230671672 Number of mis-classified points : 0.3533834586466165

# Linear Support Vector Machines
We use Linear SVM(with class balancing) because it is interpretable and works very well with high dimension data. RBF Kernel SVM is not interpretable so we cannot use it. Now we apply the training data to the model and use the CV data for finding best hyper-parameter (C)

With the best C we fit the model and test data is applied to the model. Now, the log-loss value is 1.06(near to LR) which is quite less than Random Model. Here, the total number of mis-classified cases is 36.47 percent(more than LR). Since we used class balancing we got good performance for minor classes.


Log loss : 1.116282495937613 Number of mis-classified points : 0.37030075187969924

# Random Forest Classifier


# One-hot encoding:
Normally Decision Tree works well with low-dimension data. It is also interpretable. By changing the number of base learners and max depth in Random Forest Classifier, we get best base learners=2000 and max depth=10.Then we fit the model with best hyper-parameters and test data is applied to it. The resultant log loss value is 1.1813464668656366(near to LR) and total number of mis-classified points is  41.35 percent(more than LR).

Hyper paramter tuning (With One hot Encoding)
Log loss : 1.1813464668656366 Number of mis-classified points : 0.41353383458646614

# Hyper paramter tuning (With Response Coding)

Response Coding: By changing the number of base learners and max depth in Random Forest Classifier we find that best base learners=100 and max depth=5. We then fit the model with best hyper-parameters and found that train log loss is 0.052,and CV log loss is 1.325 which says that model is overfitted even with best hyper-parameters. That is why we don’t use RF+Response Coding .

Log loss : 1.3352069773071837 Number of mis-classified points : 0.49243609022556391

# Stack the models
We stacked three classifiers — LR, SVM, NB and kept LR as the meta classifier. Now we apply the training data to the model and used the CV data for finding best hyper-parameter. With the best hyperparameter we fit the model apply the test data to the model. The log-loss value is 1.14 which is very much less than Random Model. Here we also find out that total number of mis-classified cases is 38.64 percent. Here, even though we used complex model, we got the results nearly similar to LR. Additionally, we know that the stacking classifier is not interpretable.

it is clear that with RF(Response Coding), there is a drastic change in the train and CV log loss(nearly 20 times). This means that the model is over-fitted, and thus, we remove that model. In Stacking Classifier (which is ensemble) log loss values are nearly same as LR+Balancing.  LR+Balancing suits our business or real world constraints such as interpretability and having a better log loss values than any other models.

Log loss (train) on the stacking classifier : 0.6174369409152871 Log loss (CV) on the stacking classifier : 1.112265711382749 Log loss (test) on the stacking classifier :1.1450517201676111 Number of missclassified point : 0.38646616541353385

# Maximum Voting classifier

Log loss (train) on the VotingClassifier : 0.8329702627479129 Log loss (CV) on the VotingClassifier : 1.1887678593349613 Log loss (test) on the VotingClassifier : 1.2061284826287209 Number of missclassified point : 0.3849624060150376

# Logistic regression with CountVectorizer Features, including both unigrams and bigrams
Log loss : 1.1025061826224287 Number of mis-classified points : 0.36278195488721804

# adding Variation Feature,Text Feature to improve the performance
Log loss : 0.9976654523552164 Number of mis-classified points : 0.3233082706766917
