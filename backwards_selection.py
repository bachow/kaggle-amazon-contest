'''
    __author__ = 'Brian Chow'
    __date__ = '2013-07-11'
    __Comments__ 'Modified from code by Paul Duan'
    __comments__ = 'Modified from code by Miroslaw Horbal'

    Kaggle Amazon Employee Access Challenge Logistic Regression Model with Backwards Greedy Selection
    
'''

from numpy import array, hstack
from sklearn import metrics, linear_model, preprocessing
from scipy import sparse, stats
import numpy as np
import pandas as pd
from kaggle import *

def main():
    # Options
    load_data_from_file = False

    print "Reading dataset..."

    if not load_data_from_file:
        # Skip loading unprocessed data
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))

        num_train = np.shape(train_data)[0]

        # reencode data using LabelEncoder to "normalize" to 0
        all_data = label_reencoder(all_data)

        # Transform data
        print "Transforming data..."
        dp = group_data(all_data, degree=2) 
        dt = group_data(all_data, degree=3)
        d4 = group_data(all_data, degree=4)

        y = array(train_data.ACTION)
        X = all_data[:num_train]
        X_2 = dp[:num_train]
        X_3 = dt[:num_train]
        X_4 = d4[:num_train]

        X_test = all_data[num_train:]
        X_test_2 = dp[num_train:]
        X_test_3 = dt[num_train:]
        X_test_4 = d4[num_train:]

        X_train_all = np.hstack((X, X_2, X_3, X_4))
        X_test_all = np.hstack((X_test, X_test_2, X_test_3, X_test_4))
        num_features = X_train_all.shape[1]

        # Save processed data
        np.savetxt("train_all_4.csv", np.asarray(X_train_all), delimiter = ",")
        np.savetxt("test_all_4.csv", np.asarray(X_test_all), delimiter = ",")

    else:
        ### Load already processed data from previous runs ###
        X_train_all = array(pd.read_csv('train_all.csv', header = None))
        X_test_all = array(pd.read_csv('test_all.csv', header = None))
        num_features = X_train_all.shape[1]
        train_data = pd.read_csv('train.csv')
        y = array(train_data.ACTION)
        num_train = np.shape(train_data)[0]
        num_test = np.shape(X_test_all)[0]

    Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]

    model = linear_model.LogisticRegression()

    good_features = backwards_selection(Xts, y, model, num_iter = 10)
    # good_features = [0, 7, 8, 10, 14, 24, 29, 33, 36, 37, 38, 40, 41, 42, 47, 49, 51, 53, 57, 60, 63, 64, 67, 69, 71, 79, 82]

    bestC = hyperparameter_selection(Xts, y, model, good_features)
    Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
    Xt, keymap = OneHotEncoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]

    print "Training full model..."
    model.fit(X_train, y)

    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    filename = 'logistic_backwards_preds.csv'
    create_test_submission(filename, preds)

if __name__ == "__main__":
    main()
