'''
    
    Shared classes and functions used in my Python scripts for the Amazon Employee Access Challenge on Kaggle

    __author__ = 'Brian Chow'
    __date__ = '2013-07-03'

    Some functions were written by Paul Duan and Miroslaw Horbal

    This file needs to be in the same directory as the dependent script

'''

from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model, svm, preprocessing
from scipy import sparse
from itertools import combinations, count, imap
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import multiprocessing
import bisect

SEED = 42

pool = multiprocessing.Pool(3)

def action_ranking(feature, action, sample = False, iterations = 10):
  '''
  Given lists of features and actions, returns a dictionary of features (key) and proportion of actions granted to total appearances (value). If sample = True, the function will generate a bootstrap set with replacement before calculating proportions.
  '''
  element_list = list()
  total_dict = dict()
  action_count_dict = dict()
  action_prop_dict = dict()
  count_list = [0 for i in range(10)]
  total_list = [0 for i in range(10)]
  for i in range(iterations):
    if sample:
      # create bootstrap with replacement sample set
      weight_gen = WeightedRandomGenerator([i for i in range(len(feature))])
      index = list()
      for i in range(len(feature)):
        index.append(weight_gen.next())
      for i in index:
        element_list.append((feature[i], action[i]))
    else:
      # use feature list
      for i in range(len(feature)):
        element_list.append((feature[i], action[i]))
    for element in element_list:
      if element[0] in total_dict:
        total_dict[element[0]] += 1
        action_count_dict[element[0]] += element[1]
      else:
        total_dict[element[0]] = 1
        action_count_dict[element[0]] = element[1]
  for key in total_dict:
    action_prop_dict[key] = float(action_count_dict[key]) / float(total_dict[key])
  return action_prop_dict
  # need to add some sort of cross-validation in here... K-fold stratified by ID? And then bootstrap?
  #   stratify (N = 10)
  #   for k = N - 1:
  #   fit in each fold
  #     predict
  #     calculate AUC of model against holdout fold
  # pick model with highest AUC

def backwards_selection(feature_array, results, predictive_model, num_iter = 10):
    score_hist = []
    good_features = set([f for f in range(len(feature_array))])
    removed_features = set([])
    last_dropped = None
    last_added = None
    pool = multiprocessing.Pool(3)
    # stepwise remove features until AUC does not improve
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        current_auc = 0.
        current_removed = None
        args = []
        print 'Generating arguments for cv_loop'
        for f in range(len(feature_array)):
            if f in good_features:
                feats = list(good_features)
                feats.remove(f)
                Xt = sparse.hstack([feature_array[j] for j in feats]).tocsr()
                args.append([f, Xt, results, predictive_model, num_iter])
        print 'Calculating CV'
        cycle_score = pool.map(cv_loop_mt, args)
        print 'Sorting'
        [scores.append(c) for c in cycle_score]
        current_removed = sorted(scores)[-1][1]
        good_features.remove(sorted(scores)[-1][1])
        current_auc = sorted(scores)[-1]
        last_dropped = sorted(scores)[-1][1]
        print 'Current features: {feature_list}'.format(feature_list = sorted(list(good_features)))
        print 'Best score was {best_score} attained by dropping feature {feature}'.format(feature = sorted(scores)[-1][1], best_score = sorted(scores)[-1][0])
        # stepwise add back removed features
        if len(removed_features) > 0:
            print 'Adding back removed features and testing'
            args = []
            scores = []
            for rf in removed_features:
                feats = list(good_features) + [rf]
                Xt = sparse.hstack([feature_array[j] for j in feats]).tocsr()
                args.append([rf, Xt, results, predictive_model, num_iter])
            cycle_score = pool.map(cv_loop_mt, args)
            [scores.append(c) for c in cycle_score]
            if (sorted(scores)[-1][0]) > current_auc[0]:
                removed_features.remove(sorted(scores)[-1][1])
                good_features.add(sorted(scores)[-1][1])
                print 'Current features: {feature_list}'.format(feature_list = sorted(list(good_features)))
                print 'Best score was {best_score} attained by adding feature {feature}'.format(feature = sorted(scores)[-1][1], best_score = sorted(scores)[-1][0])
                score_hist.append(sorted(scores)[-1])
                last_added = sorted(scores)[-1][1]
            else:
                print 'Did not add any previously removed features'
                score_hist.append(current_auc)
                last_added = None
        removed_features.add(current_removed)
    pool.terminate()
    good_features.add(last_dropped)
    if last_added != None:
        good_features.remove(last_added)
    good_features = sorted(list(good_features))
    return good_features

def cantor_hash(combo):
    '''
    Takes a list of length N and applies Cantor pairing function to produce a unique value
    '''
    combo_length = len(combo)
    cantor = 0
    if combo_length == 2:
        return 0.5 * (combo[0] + combo[1]) * (combo[0] + combo[1] + 1) + combo[1]
    elif combo_length == 3:
        interim_val = 0.5 * (combo[0] + combo[1]) * (combo[0] + combo[1] + 1) + combo[1]
        return 0.5 * (interim_val + combo[2]) * (interim_val + combo[2] + 1) + combo[2]
    elif combo_length == 4:
        interim_val1 = 0.5 * (combo[0] + combo[1]) * (combo[0] + combo[1] + 1) + combo[1]
        interim_val2 = 0.5 * (combo[2] + combo[3]) * (combo[2] + combo[3] + 1) + combo[3]
        return 0.5 * (interim_val1 + interim_val2) * (interim_val1 + interim_val2 + 1) + interim_val2

def count_occurrence(listobj):
  '''
  Count occurrences of unique identifiers
  Given a list of objects, returns a dictionary of counts by unique object
  '''
  elements = dict()
  for item in listobj:
    if item in elements:
      elements[item] += 1
    else:
      elements[item] = 1
  return elements

def create_cutoffs(feature_array, cutoffs = [20, 10, 5, 4, 3, 2]):
    '''
        Given a 1-D array of categories, returns an array of length <= len(cutoffs) that contains category IDs (of same sorting as original array) that exceed the cutoff count by: max count / interval.
    '''
    ### function code ###
    # declare listobj to store cutoff category ids
    Xts = [[] for j in cutoffs]
    # create cutoff category ids
    for j in range(len(cutoffs)):
        # returns vector of categories over/under cutoff (0 and 1, respectively), in order of categories in feature array
        cutoff_vector = rare_event_detection(feature_array, cutoff = cutoffs[j])
        # if vector is 1, return -1; else, return category id
        for k in range(len(feature_array)):
            if cutoff_vector[k] == 1:
                Xts[j].append(-1)
            else:
                Xts[j].append(feature_array[k])
    return array(Xts)

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        if len(np.shape(X_train)) == 1:
            trainrows = np.shape(X_train)[0]
            cvrows = np.shape(X_cv)[0]
            model.fit(X_train.reshape(trainrows, 1), y_train)
            preds = model.predict_proba(X_cv.reshape(cvrows,1))[:,1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        print 'AUC (fold {current}/{total}): {auc}'.format(current = i + 1, total = N, auc = auc)
        mean_auc += auc
    return mean_auc/N

def cv_loop_cut(args):
    f, cut_iter, X, y, model, N = args
    sum_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        print 'Feature Set {f} (Cut {c}) - AUC (fold {current}/{total}): {auc}'.format(f = f, c = cut_iter, current = i + 1, total = N, auc = auc)
        sum_auc += auc
    mean_auc = sum_auc / N
    return (mean_auc, f, cut_iter)

def cv_loop_mt(args):
    f, X, y, model, N = args
    sum_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        print 'Feature Set {f} AUC (fold {current}/{total}): {auc}'.format(f = f, current = i + 1, total = N, auc = auc)
        sum_auc += auc
    mean_auc = sum_auc / N
    return (mean_auc, f)

def cv_loop_multi(args):
  f, X, y, model, N = args
  return (cv_loop(X, y, model, N), f)

def forward_selection_2(feature_array, results, predictive_model, num_iter = 10, pool = pool):
    score_hist = []
    good_features = set([])
    last_dropped = None
    last_added = None
    # stepwise remove features until AUC does not improve
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        current_auc = 0.
        args = []
        print 'Generating arguments for cv_loop'
        for f in range(len(feature_array)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = sparse.hstack([feature_array[j] for j in feats]).tocsr()
                args.append([f, Xt, results, predictive_model, num_iter])
        print 'Calculating CV'
        cycle_score = pool.map(cv_loop_mt, args)
        print 'Sorting'
        [scores.append(c) for c in cycle_score]
        last_added = sorted(scores)[-1]
        good_features.add(last_added[1])
        print 'Current features: {feature_list}'.format(feature_list = sorted(list(good_features)))
        print 'Best score was {best_score} attained by adding feature {feature}'.format(feature = last_added[1], best_score = last_added[0])
        # stepwise add back removed features
        if len(good_features) > 1:
            print 'Stepwise removing good_features and testing'
            args = []
            scores = []
            for f in good_features:
                if f != last_added[1]:
                    feats = list(good_features)
                    feats.remove(f)
                    Xt = sparse.hstack([feature_array[j] for j in feats]).tocsr()
                    args.append([f, Xt, results, predictive_model, num_iter])
            cycle_score = pool.map(cv_loop_mt, args)
            [scores.append(c) for c in cycle_score]
            if (sorted(scores)[-1][0]) > last_added[0]:
                good_features.remove(sorted(scores)[-1][1])
                print 'Current features: {feature_list}'.format(feature_list = sorted(list(good_features)))
                print 'Best score was {best_score} attained by dropping feature {feature}'.format(feature = sorted(scores)[-1][1], best_score = sorted(scores)[-1][0])
                score_hist.append(sorted(scores)[-1])
                last_dropped = sorted(scores)[-1][1]
            else:
                print 'Did not drop any previously added features'
                score_hist.append(last_added)
                last_dropped = None
    good_features.remove(last_added[1])
    if last_dropped != None:
        good_features.add(last_dropped)
    good_features = sorted(list(good_features))
    for i in score_hist:
        print i
    return good_features

def greedy_selection(feature_array, results, predictive_model, num_iter = 10):
    score_hist = []
    good_features = set([])
    # Greedy feature selection loop
    pool = multiprocessing.Pool(3)
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        args = []
        for f in range(len(feature_array)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = sparse.hstack([feature_array[j] for j in feats]).tocsr()
                args.append([f, Xt, results, predictive_model, num_iter])
        cycle_score = pool.map(cv_loop_multi, args)
        for c in cycle_score:
            scores.append(c)
        good_features.add(sorted(scores)[-1][1])
        score_hist.append(sorted(scores)[-1])
        print "Current features: %s" % sorted(list(good_features))
        print "Best score was: %f" % (sorted(scores)[-1][1])
    # terminate spawned processes
    pool.terminate()
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))
    return good_features

def greedy_selection_cutoff(cutoff_array, results, predictive_model, num_iter = 5):
    score_hist = []
    good_features = set([])
    num_feats = len(cutoff_array)
    # Greedy feature selection loop
    good_feats_array = array([])
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        args = []
        for f in range(num_feats):
            if f not in good_features:
                cut_iter = 0
                for cut in cutoff_array[f]:
                    m = np.shape(good_feats_array)[0]
                    if m != 0:
                        Xt = sparse.hstack([good_feats_array, cut]).tocsr()
                    else:
                        Xt = cut
                    args.append([f, cut_iter, Xt, results, predictive_model, num_iter])
                    cut_iter += 1
        cycle_score = pool.map(cv_loop_cut, args)
        for c in cycle_score:
            scores.append(c)
        auc, feat, cut = sorted(scores)[-1]
        good_features.add(feat)
        score_hist.append(sorted(scores)[-1])
        m = np.shape(good_feats_array)[0]
        if m != 0:
            good_feats_array = sparse.hstack([good_feats_array, cutoff_array[feat][cut]]).tocsr()
        else:
            good_feats_array = cutoff_array[feat][cut]
        print "Current features: {}".format(sorted(list(good_features)))
        print "Best score was: {auc} by adding feature {feat}, cut {cut}".format(auc = auc, feat = feat, cut = cut)
    # remove last added feature
    good_features.remove(feat)
    good_features = sorted(list(good_features))
    return good_features, score_hist

def greedy_selection_nonsparse(feature_array, results, predictive_model, num_iter = 10, SEED = 42, pool = pool):
    ### New greedy selection algorithm that works with non-sparse data ###
    score_hist = []
    good_features = set([])
    # Greedy feature selection loop
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        args = []
        for f in range(len(feature_array)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = np.vstack([feature_array[j,] for j in feats]).T
                args.append([f, Xt, results, predictive_model, num_iter])
        cycle_score = pool.map(cv_loop_mt, args)
        for c in cycle_score:
            scores.append(c)
        good_features.add(sorted(scores)[-1][1])
        score_hist.append(sorted(scores)[-1])
        print "Current features: %s" % sorted(list(good_features))
        print "Best score was: %f" % (sorted(scores)[-1][0])
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))
    return good_features

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([cantor_hash(combo) for combo in data[:,indicies]])
    return array(new_data).T

def hyperparameter_selection(feature_array, results, predictive_model, good_features, num_iter = 10):
    '''
    hyperparameter selection loop.
    INPUT: array of features, array of results, predictive model, and set of good features
    OUTPUT: best hyperparameter C value
    '''
    score_hist = []
    sparse_feature_array = sparse.hstack([feature_array[j] for j in good_features]).tocsr()
    Cvals = np.logspace(-4, 4, 15, base=2)
    for C in Cvals:
        predictive_model.C = C
        score = cv_loop(sparse_feature_array, results, predictive_model, num_iter)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = sorted(score_hist)[-1][1]
    print "Best C value (lm): %f" % (bestC)
    predictive_model.C = bestC
    return bestC

def hyperparameter_selection_2(feature_array, results, predictive_model, num_iter = 10):
    '''
    hyperparameter selection loop.
    INPUT: array of features, array of results, predictive model, and set of good features
    OUTPUT: best hyperparameter C value
    '''
    score_hist = []
    Cvals = np.logspace(-4, 4, 15, base=2)
    for C in Cvals:
        predictive_model.C = C
        score = cv_loop(feature_array, results, predictive_model, num_iter)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = sorted(score_hist)[-1][1]
    print "Best C value (lm): %f" % (bestC)
    predictive_model.C = bestC
    return bestC

def label_reencoder(listobj):
    '''
    Reencodes categorical data to "normalize" smallest value to 0
    '''
    le = preprocessing.LabelEncoder()
    reencoded = []
    for row in listobj.T:
        le.fit(row)
        reencoded.append(le.transform(row))
    return array(reencoded).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
        keymap = []
        for col in data.T:
            uniques = set(list(col))
            keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
        km = keymap[i]
        num_labels = len(km)
        spmat = sparse.lil_matrix((total_pts, num_labels))
        for j, val in enumerate(col):
            if val in km:
                spmat[j, km[val]] = 1
        outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def pick_top_auc(feature_array, results, predictive_model, num_iter = 10, pool = pool):
    scores = []
    args = []
    for f in range(len(feature_array)):
        Xt = feature_array[f]
        args.append([f, Xt, results, predictive_model, num_iter])
    cycle_score = pool.map(cv_loop_mt, args)
    for c in cycle_score:
        scores.append(c)
    good_features = sorted(scores)[-1][1]
    print "Best cutoff: %s" % (sorted(scores)[-1][1])
    print "Best score was: %f" % (sorted(scores)[-1][0])
    return good_features

def rare_event_detection(objarray, cutoff = 3):
  '''
  Creates dictionary of category - count pairs. For categories in a feature, if it is below the cutoff, returns a 1 (otherwise 0). Function returns an array of categories below cutoff in original order of entry.
  '''
  if len(np.shape(objarray)) > 1:
    m, n = np.shape(objarray)
  else:
    n = 1
  for colindex in range(n):
    if n > 1:
      rare_events = [[] for i in range(n)]
      occur_list = count_occurrence(objarray[:, colindex])
      for item in objarray[:, colindex]:
        if occur_list[item] < cutoff:
          rare_events[colindex].append(1)
        else:
          rare_events[colindex].append(0)
    else:
      rare_events = []
      occur_list = count_occurrence(objarray)
      for item in objarray:
        if occur_list[item] < cutoff:
          rare_events.append(1)
        else:
          rare_events.append(0)
  return array(rare_events)

class WeightedRandomGenerator(object):
  def __init__(self, weights):
      self.totals = []
      running_total = 0
      for w in weights:
          running_total += w
          self.totals.append(running_total)
  def next(self):
      rnd = random.random() * self.totals[-1]
      return bisect.bisect_right(self.totals, rnd)
  def __call__(self):
      return self.next()

pool.terminate()
pool = multiprocessing.Pool(3)
