kaggle-amazon-contest
=====================
This repository contains my various Python scripts that I wrote for the Amazon Employee Access Challenge on Kaggle.com.

The training and testing data can be downloaded from http://www.kaggle.com/c/amazon-employee-access-challenge


Short Descriptions of Scripts
=============================
kaggle.py   :: Contains shared functions and classes upon which other scripts depend. This must be in the same directory as the other scripts
              
boost.py    :: Ensemble learner based on an ADABoost implementation. Initially, the learner creates a bootstrap with replacement data set from the training set upon which to train a logistic model. The learner finds misclassified entries and creates weights for subsequent bootstrapping. This process is iterated N times. All predictions are collected and an ensemble model is fit to the predictions to create weights for model averaging.
