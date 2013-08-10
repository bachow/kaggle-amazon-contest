'''

    __author__ = 'Brian Chow'
    __date__ = '2013-07-29'

    4 degree and rare event feature engineering, fit to logistic regression model

'''
from numpy import array, hstack
from sklearn import linear_model
from scipy import sparse, stats
from kaggle import *
import numpy as np
import pandas as pd

SEED = 42

def main():
    # Options
    load_data_from_file = False
    processed_train_data_file = 'train_all.csv'
    processed_test_data_file = 'test_all.csv'
    train_data = 'train.csv'
    test_data = 'test.csv'
    preds_file = 'cutoffs_preds.csv'

    if not load_data_from_file:
        ### Load and Process Data ###
        train_data = pd.read_csv(train_data)
        test_data = pd.read_csv(test_data)
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

        # Compile data
        X_train_all = np.hstack((X, X_2, X_3, X_4))
        X_test_all = np.hstack((X_test, X_test_2, X_test_3, X_test_4))
        num_features = X_train_all.shape[1]
        num_test = np.shape(X_test_all)[0]

        # Save processed data
        np.savetxt("train_all.csv", np.asarray(X_train_all), delimiter = ",")
        np.savetxt("test_all.csv", np.asarray(X_test_all), delimiter = ",")

    else:
        ### Load already processed data from previous runs ###
        X_train_all = array(pd.read_csv(processed_train_data_file, header = None))
        X_test_all = array(pd.read_csv(processed_test_data_file, header = None))
        train_data = pd.read_csv(train_data)
        y = array(train_data.ACTION)
        num_train = np.shape(train_data)[0]
        num_test = np.shape(X_test_all)[0]
        num_features = X_train_all.shape[1]

    ### Create index of cutoff data for use in feature selection ###
    cutoffs = []
    for i in range(num_features):
        print 'Creating Index of Cutoffs for Feature ', i
        cutoffs.append(create_cutoffs(X_train_all[:,i]))

    # # debug
    # sum_size = 0
    # for i in cutoffs:
    #     sum_size += np.shape(i)[0]
    #     print np.shape(i)[0]
    # sum_size # = 1058

    ### One hot encode cutoffs ###
    cos = [[] for i in range(num_features)]
    # cutoffs are ordered as cos[feature #][cutoff #] and encoded in CSR
    for i in range(num_features):
        print 'One Hot Encoding Feature', i
        cos[i].append([OneHotEncoder(array([cutoffs[i][j]]).T)[0] for j in range(len(cutoffs[i]))])
    # gotta remove extraneous lists
    for i in range(num_features):
        cos[i] = cos[i][0]

    ### Declaring model ###
    model = linear_model.LogisticRegression()

    ### Calculate good features and cuts ###
    good_features, score_hist = greedy_selection_cutoff(cos, y, model)

    ### get ordered lists of good features and respective cuts ###
    ordered_good_cuts = []
    ordered_features = []
    for i in sorted(score_hist):
        ordered_good_cuts.append(i[2])
        ordered_features.append(i[1])

    good_features = sorted(ordered_features)

    ### Create sparse matrix of only good features and cuts for hyperparameter selection ###
    cos_gf = []
    for i in range(len(ordered_features)):
        cos_gf.append([OneHotEncoder(array([cutoffs[ordered_features[i]][ordered_good_cuts[i]]]).T)[0]])

    for i in range(len(cos_gf)):
        cos_gf[i] = cos_gf[i][0]

    cos_gfs = sparse.hstack(cos_gf).tocsr()

    ### Hyperparameter selection for good features and cuts ###
    bestC = hyperparameter_selection_2(cos_gfs, y, model)

    ### Create test matrix with good features and cutoffs ###
    train_cutoffs = []
    test_cutoffs = []
    for i in range(len(ordered_features)):
        print 'Feature', i
        train_cutoffs.append(create_cutoffs(X_train_all[:,ordered_features[i]], cutoffs = [ordered_good_cuts[i]]))
        test_cutoffs.append(create_cutoffs(X_test_all[:,ordered_features[i]], cutoffs = [ordered_good_cuts[i]]))

    for i in range(len(train_cutoffs)):
        train_cutoffs[i] = train_cutoffs[i][0]
        test_cutoffs[i] = test_cutoffs[i][0]

    Xt = np.hstack((train_cutoffs, test_cutoffs)).T
    Xt, keymap = OneHotEncoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]

    print "Training full model..."
    model.fit(X_train, y)

    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    create_test_submission(preds_file, preds)
    # scored 0.89912 on Kaggle

if __name__ == "__main__":
    main()
