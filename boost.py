'''
    __author__ = 'Brian Chow'
    __date__ = '2013-06-26'
    __Comments__ 'Modified from code by Paul Duan'
    __comments__ = 'Modified from code by Miroslaw Horbal (miroslaw@gmail.com)

    Kaggle Amazon Employee Access Challenge Ensemble Logistic Regression Model constructed using an ADABOOST implementation
'''

from numpy import array, hstack
from sklearn import metrics, linear_model, preprocessing
from scipy import sparse
import numpy as np
import pandas as pd
import multiprocessing
from kaggle import *

SEED = 42
    
def main():    
    '''
    Comments
    Good for AUC = 0.847
    '''

    # Options
    print_all = False
    print_preds = True
    print_preds_file = 'ensemble_predictions_'
    print_processed_data = True
    print_processed_data_file = 'ensemble_iteration_'
    print_good_features = True
    print_good_features_file = 'good_features_'
    print_weights = True
    print_weights_file = 'ensemble_weights_'
    print_bootstrap_indicies = True
    print_bootstrap_indicies_file = 'ensemble_bootstrap_indices_'
    load_data_from_file = False
    processed_train_data_file = 'train_all.csv'
    processed_test_data_file = 'test_all.csv'
    train_data = 'train.csv'
    test_data = 'test.csv'
    do_bootstrap = True

    print "Reading dataset..."

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

        y = array(train_data.ACTION)
        X = all_data[:num_train]
        X_2 = dp[:num_train]
        X_3 = dt[:num_train]

        X_test = all_data[num_train:]
        X_test_2 = dp[num_train:]
        X_test_3 = dt[num_train:]

        # Compile data
        X_train_all = np.hstack((X, X_2, X_3))
        X_test_all = np.hstack((X_test, X_test_2, X_test_3))
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

    ### ADABOOST ###

    ### Initialize variables for Boost ###
    Dtx = [[1.0 / num_train for i in range(num_train)]]
    bootstrap_index = []
    good_features = []
    bestC = []
    lm_preds_train = []
    lm_preds_test = []
    lm_preds_bin = []
    selection_weight_normal = 1.0 / num_train
    auc_hist = []
    iteration = 0
    lm_models = list()
    max_iterations = 20

    pool = multiprocessing.Pool(3)

    while len(auc_hist) < max_iterations:
        ''' ADABoost loop '''
        # load models
        print "Loading model"
        lm_models.append(linear_model.LogisticRegression())
        # Bootstrap training data with weights
        print "Generating Bootstrap Data"
        index = []
        weight_gen = WeightedRandomGenerator(Dtx[iteration])
        for i in range(len(Dtx[iteration])):
            index.append(weight_gen.next())
        bootstrap_index.append(index)
        X_train_boot = array([X_train_all[v] for v in bootstrap_index[iteration]])
        y_boot = array([y[v] for v in bootstrap_index[iteration]])
        print "One Hot encoding features"
        Xts_boot = [OneHotEncoder(X_train_boot[:,[i]])[0] for i in range(num_features)]
        # Greedy feature selection and hyperparameter optimization on bootstrap data
        print "Running Greedy Forward Selection"
        good_features.append(greedy_selection(Xts_boot, y_boot, lm_models[iteration]))
        filename = 'features_iteration_' + str(iteration) + '.csv'
        create_test_submission(filename, good_features[iteration])
        print "Running Hyperparameter Optimization"
        bestC.append(hyperparameter_selection(Xts_boot, y_boot, lm_models[iteration], good_features[iteration]))
        lm_models[iteration].C = bestC[iteration]
        # Cross-validation of model on full data set (bootstrap + test)
        print "Performing One Hot Encoding on entire dataset with good features..."
        Xt = np.vstack((X_train_boot[:,good_features[iteration]], X_test_all[:,good_features[iteration]], X_train_all[:,good_features[iteration]]))
        Xt, keymap = OneHotEncoder(Xt)
        X_train_boot_sparse = Xt[:num_train]
        X_test_sparse = Xt[num_train:num_test+num_train]
        X_train_sparse = Xt[num_test+num_train:]
        print "Training full model..."
        lm_models[iteration].fit(X_train_boot_sparse, y_boot)
        # Predict results
        lm_preds_bin.append(lm_models[iteration].predict(X_train_sparse))
        lm_preds_train.append(lm_models[iteration].predict_proba(X_train_sparse)[:,1])
        lm_preds_test.append(lm_models[iteration].predict_proba(X_test_sparse)[:,1])
        # Identify misclassified data and calculate weights
        pred_y = lm_preds_bin[iteration]
        misclassed = 0
        epsilon_t = 0.
        for i in range(len(pred_y)):
            if pred_y[i] != y[i]:
                misclassed += 1
                epsilon_t += Dtx[iteration][i]
        beta_t = epsilon_t / (1 - epsilon_t)
        Dtx.append(list(Dtx[-1]))
        for i in range(len(Dtx[iteration])):
            if pred_y[i] == y[i]:
                Dtx[iteration + 1][i] *= beta_t
        total = sum(weight for weight in Dtx[iteration + 1])
        Dtx[iteration + 1] = [v / total for v in Dtx[iteration + 1]]
        # calculate AUC for this model
        auc_hist.append(metrics.auc_score(y, pred_y))
        print "AUC for this solution: ", auc_hist[iteration]

        # print data
        if print_bootstrap_indicies:
            filename = print_bootstrap_indicies_file + str(iteration) + '.csv'
            create_test_submission(filename, bootstrap_index[iteration])

        if print_good_features:
            filename = print_good_features_file + str(iteration) + '.csv'
            create_test_submission(filename, good_features[iteration])

        if print_preds:
            filename = print_preds_file + str(iteration) + '.csv'
            create_test_submission(filename, lm_preds_test[iteration])

        if print_weights:
            filename = print_weights_file + str(iteration) + '.csv'
            create_test_submission(filename, Dtx[iteration + 1])

        # Iterate
        iteration += 1

    # stack predictions
    lm_preds_bin2 = lm_preds_bin[0]
    for i in range(1, len(lm_preds_bin)):
        lm_preds_bin2 = np.vstack((lm_preds_bin2, lm_preds_bin[i]))

    lm_preds_bin2 = lm_preds_bin2.T

    # fit a logistic model on predictions against y and extract coefficients
    ensemble_model = linear_model.LogisticRegression()
    ensemble_model.fit(lm_preds_bin2, y)
    ensemble_coefs = ensemble_model.coef_

    # forgot how to force positive coefs... set all negative coefs to 0
    for i in range(len(ensemble_coefs[0])):
        if ensemble_coefs[0][i] < 0:
            ensemble_coefs[0][i] = 0

    # weigh predictions by coefs and generate weighted average preds
    coefs_total = sum(ensemble_coefs[0])
    ensemble_preds = []
    for i in range(num_train):
        mean_pred = 0.
        for j in range(len(ensemble_coefs[0])):
            mean_pred += lm_preds_train[j][i] * ensemble_coefs[0][j]
        mean_pred = mean_pred / coefs_total
        ensemble_preds.append(mean_pred)

    # do the above for test predictions
    lm_preds_test2 = lm_preds_test[0]
    for i in range(1, len(lm_preds_test)):
        lm_preds_test2 = np.vstack((lm_preds_test2, lm_preds_test[i]))

    coefs_total = sum(ensemble_coefs[0])
    ensemble_test_preds = []
    for i in range(num_test):
        mean_pred = 0.
        for j in range(len(ensemble_coefs[0])):
            mean_pred += lm_preds_test2[j][i] * ensemble_coefs[0][j]
        mean_pred = mean_pred / coefs_total
        ensemble_test_preds.append(mean_pred)

    # output ensemble test predictions
    filename = '2013-07-27_ensemble_preds.csv'
    create_test_submission(filename, ensemble_test_preds)
    # scored 0.90128 on Kaggle
    
if __name__ == "__main__":
    main()
