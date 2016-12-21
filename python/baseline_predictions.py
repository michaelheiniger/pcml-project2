import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import helpers as h

########################################################################
# Global mean baseline
########################################################################
def run_cross_validation_global_baseline(ratings, k_fold):
    """cross-validation for the global baseline
    input:
        ratings: the given ratings
        k_fold: the number of folds
    output:
        rmse_tr: the train errors for every fold
        rmse_te: the test errors for every fold
    """
    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, 1))
    rmse_te = np.zeros((k_fold, 1))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        train_ratings, test_ratings = get_data_sets_for_cross_validation(ratings, k_indices, k)
        rmse_tr[k], rmse_te[k] = baseline_global_mean(train_ratings, test_ratings)

    return rmse_tr, rmse_te

def baseline_global_mean(train, test):
    """baseline using the global mean.
    input: 
        train: the training set
        test: the test set
    output:
        rmse_tr: rmse on the training set
        rmse_te: rmse on the test set
    """

    _, _, values_train = sp.find(train)

    # Compute the global mean
    global_mean = sum(values_train) / len(values_train)

    # Compute RMSE for training
    mse_tr = 0
    for k in range(0, len(values_train)):
        mse_tr += np.power(values_train[k] - global_mean, 2)

    # Normalize
    mse_tr /= len(values_train)

    # Compute RMSE for  test
    _, _, values_test = sp.find(test)
    mse_te = 0
    for k in range(0, len(values_test)):
        mse_te += np.power(values_test[k] - global_mean, 2)

    # Normalize
    mse_te /= len(values_test)

    # Compute RMSE
    rmse_tr = np.sqrt(2 * mse_tr)
    rmse_te = np.sqrt(2 * mse_te)

    return rmse_tr, rmse_te


########################################################################
# User mean baseline
########################################################################
def run_cross_validation_user_baseline(ratings, k_fold):
    """cross-validation for user mean baseline
    input:
        ratings: the given ratings
        k_fold: the number of folds
    output:
        rmse_tr: the train errors for every fold
        rmse_te: the test errors for every fold
    """
    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, 1))
    rmse_te = np.zeros((k_fold, 1))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        train_ratings, test_ratings = get_data_sets_for_cross_validation(ratings, k_indices, k)
        rmse_tr[k], rmse_te[k] = baseline_user_mean(train_ratings, test_ratings)

    return rmse_tr, rmse_te

def baseline_user_mean(train, test):
    """baseline using the user means as the prediction.
    input: 
        train: the training set
        test: the test set
    output:
        rmse_tr: rmse on the training set
        rmse_te: rmse on the test set
    """
    num_items, num_users = train.shape

    # Compute mean for each user
    _, j_tr, v_tr = sp.find(train)
    countings = np.bincount(j_tr)
    sums_tr = np.bincount(j_tr, weights=v_tr)
    mean_users = sums_tr / countings

    # Compute MSE for train
    mse_tr = 0

    # Iterate over all user ratings in train
    for i in range(0, len(j_tr)):
        mse_tr += np.power(v_tr[i] - mean_users[j_tr[i]], 2)

    # Normalize
    mse_tr /= len(v_tr)

    # Compute MSE for test
    mse_te = 0
    _, j_te, v_te = sp.find(test)

    # Iterate over all user ratings in test
    for i in range(0, len(j_te)):
        mse_te += np.power(v_te[i] - mean_users[j_te[i]], 2)

    # Normalize
    mse_te /= len(v_te)

    rmse_tr = np.sqrt(2 * mse_tr)
    rmse_te = np.sqrt(2 * mse_te)

    return rmse_tr, rmse_te


########################################################################
# Item mean baseline
########################################################################
def run_cross_validation_item_baseline(ratings, k_fold):
    """cross-validation for the global baseline
    input:
        ratings: the given ratings
        k_fold: the number of folds
    output:
        rmse_tr: the train errors for every fold
        rmse_te: the test errors for every fold
    """
    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, 1))
    rmse_te = np.zeros((k_fold, 1))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        train_ratings, test_ratings = get_data_sets_for_cross_validation(ratings, k_indices, k)
        rmse_tr[k], rmse_te[k] = baseline_item_mean(train_ratings, test_ratings)

    return rmse_tr, rmse_te


def baseline_item_mean(train, test):
    """baseline using item means as the prediction.
    input: 
        train: the training set
        test: the test set
    output:
        rmse_tr: rmse on the training set
        rmse_te: rmse on the test set
    """

    num_items, num_users = train.shape

    # Compute mean for each movie
    i_tr, _, v_tr = sp.find(train)
    countings = np.bincount(i_tr)
    sums_tr = np.bincount(i_tr, weights=v_tr)
    mean_items = sums_tr / countings

    # Compute MSE train
    mse_tr = 0

    # Iterate over all movie ratings
    for i in range(0, len(i_tr)):
        mse_tr += np.power(v_tr[i] - mean_items[i_tr[i]], 2)

    # Normalize
    mse_tr /= len(i_tr)

    # Compute MSE test
    mse_te = 0

    i_te, _, v_te = sp.find(test)

    # Iterate over all movie ratings
    for i in range(0, len(i_te)):
        mse_te += np.power(v_te[i] - mean_items[i_te[i]], 2)

    # Normalize
    mse_te /= len(i_te)

    # Compute RMSE
    rmse_tr = np.sqrt(2 * mse_tr)
    rmse_te = np.sqrt(2 * mse_te)

    return rmse_tr, rmse_te


############################################################
# Form the K folds for cross-validation
############################################################
def get_data_sets_for_cross_validation(ratings, k_indices, k):
    """splits the ratings into a train and a test set
    input: 
        ratings: the given ratings
        k_indices: the indices for k_fold crossvalidation
        k: the number of the fold"""

    indices_test = k_indices[k]
    indices_train = np.delete(k_indices.reshape(k_indices.size, 1), indices_test, axis=0).squeeze(axis=1)

    I, J, V = sp.find(ratings)

    I_train = I[indices_train]
    J_train = J[indices_train]
    V_train = V[indices_train]

    I_test = I[indices_test]
    J_test = J[indices_test]
    V_test = V[indices_test]

    train_ratings = sp.lil_matrix(sp.coo_matrix((V_train, (I_train, J_train)), (ratings.shape)))
    test_ratings = sp.lil_matrix(sp.coo_matrix((V_test, (I_test, J_test)), (ratings.shape)))

    return train_ratings, test_ratings