kaggle-amazon-contest
=====================
This repository contains my various Python scripts that I wrote for the Amazon Employee Access Challenge on Kaggle.com.

The training and testing data can be downloaded from http://www.kaggle.com/c/amazon-employee-access-challenge


Short Descriptions of Scripts
=============================
kaggle.py
:: Contains shared functions and classes upon which other scripts depend. This must be in the same directory as the other scripts
              
boost.py
:: Ensemble learner based on an ADABoost implementation. Initially, the learner creates a bootstrap with replacement data set from the training set upon which to train a logistic model. The learner finds misclassified entries and creates weights for subsequent bootstrapping. This process is iterated N times. All predictions are collected and an ensemble model is fit to the predictions to create weights for model averaging.

backwards_selection.py      
:: This is essentially the same as Miroslaw Horbal's logistic regression code (http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/4838/python-code-to-achieve-0-90-auc-with-logistic-regression), but I've implemented a backwards greedy selection model that also checks to see if previously dropped features increase AUC when added back to the model.

cutoffs.py
:: Lots of feature engineering: 4th degree combinations of features and rare feature event cutoffs. For the latter, occurrences of levels of features were counted, and arbitrary cutoffs were introduced. The best cutoffs were determined by cross-validation of AUC scores, and these were included in the final logistic regression models.
