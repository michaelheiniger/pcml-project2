import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

import helpers as h
import plots as p


def init_MF(train, num_features):
    """ Init the parameter for matrix factorization.
    input:
        train: the train dataset
        num_features: the dimension (K) of the subspace
    output:
        user_features: initialized user feature matrix
        item_features: initialized item feature matrix
    """

    num_items, num_users = train.shape

    # Init matrices at random
    user_features = np.random.random((num_features, num_users))
    item_features = np.random.random((num_features, num_items))

    return user_features, item_features


def compute_error(data, W, Z, nz):
    """ Compute the loss (RMSE) of the prediction of nonzero elements.
    input:
        data: the known ratings
        W: item feature matrix
        Z: user feature matrix
        nz: nonzero indices of data
    output:
        rmse: the RMSE of the predicted elements"""

    wz = np.transpose(prediction(W, Z))

    mse = 0
    for d, n in nz:
        mse += np.power(data[d, n] - wz[d, n], 2)
    mse = mse / len(nz)
    rmse = np.sqrt(mse)  # The factor 2 disappears because of the 1/2 of MSE

    return rmse


def prediction(W, Z):
    """ Compute the predicted matrix W.dot(Z.T) of size DxN for W (KxD) and Z (KxN) 
    input:
        W: item feature matrix
        Z: user feature matrix
    ouput:
        the prediction of the ratings
        """

    return np.dot(W.T, Z)


def mf_gd_regularized(train, test, num_epochs, gamma, num_features, lambda_user, lambda_item):
    """ Matrix factorization using GD 
    input: 
        train: the train data
        test: the test data
        gamma: the initial learning rate
        num_epochs: number iterations of algorithm
        num_features: the dimension (K) of the subspace
        lambda_user: regularization parameter for the user feature matrix
        lambda_item: regularization parameter for the item feature matrix
    output:
        rmse_train: train errors for every epoch
        rmse_test: test_errors for every epoch
        """

    # Init basis (W) and coeff (Z) matrices
    Z, W = init_MF(train, num_features)

    # Find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    # Store RMSE for each epochs
    rmse_train = np.zeros((num_epochs, 1))
    rmse_test = np.zeros((num_epochs, 1))

    print("Learn the matrix factorization using GD...")
    for it in range(num_epochs):
        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            e = train[d, n] - prediction(W[:, d], Z[:, n])
            Z[:, n] += gamma * (e * W[:, d] - lambda_user * Z[:, n])
            W[:, d] += gamma * (e * Z[:, n] - lambda_item * W[:, d])

        nz_row, nz_col = train.nonzero()
        nz_train = list(zip(nz_row, nz_col))
        rmse_train[it] = compute_error(train, Z, W, nz_train)
        rmse_test[it] = compute_error(test, Z, W, nz_test)
        # print("iter: {}, RMSE on training set: {}.".format(it, rmse_train))
        # print("RMSE on test data: {}.".format(rmse_test))

    return rmse_train, rmse_test


def cross_validation(ratings, k_indices, k, num_epochs, gamma, num_features, lambda_user, lambda_item):
    """ Perform K-Fold Cross-validation for Matrix Factorization with GD
    input: 
        ratings: the dataset
        k_indices: indices for the k_fold crossvalidaton
        k: number of the fold
        num_epochs: number iterations of algorithm
        gamma: the initial learning rate
        num_features: the dimension (K) of the subspace
        lambda_user: regularization parameter for the user feature matrix
        lambda_item: regularization parameter for the item feature matrix
    output:
        final_rmse_train: train rmse of last epoch
        final_rmse_test: test rmse of last epoch
        """

    ############################################################
    # Form the K folds
    ############################################################
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

    ############################################################
    # Matrix Factorization (using Stochastic Gradient Descent)
    ############################################################
    # Returned RMSE are arrays with rmse of each epochs
    rmse_train, rmse_test = mf_gd_regularized(train_ratings, test_ratings, num_epochs, gamma, num_features,
                                               lambda_user, lambda_item)

    # Final RMSE is the one of last epoch
    final_rmse_train = rmse_train[-1]
    final_rmse_test = rmse_test[-1]

    return final_rmse_train, final_rmse_test


def run_mf_cv_num_features(ratings, k_fold, num_epochs, num_features, lambda_user, lambda_item):
    """ Performs cross-validation with variable number of features 
    input: 
        ratings: the dataset
        k_fold: number of the folds
        num_epochs: number iterations of algorithm
        num_features: array of values for the dimension (K) of the subspace
        lambda_user: regularization parameter for the user feature matrix
        lambda_item: regularization parameter for the item feature matrix
    output:
        rmse_tr:  matrix of train errrors with shape (k_fold, len(num_features))
        rmse_te:  matrix of test errors with shape (k_fold, len(num_features))"""

    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(num_features)))
    rmse_te = np.zeros((k_fold, len(num_features)))

    gamma = 0.01

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for i, n_features in enumerate(num_features):
            rmse_tr[k, i], rmse_te[k, i] = cross_validation(ratings, k_indices, k, num_epochs, gamma, n_features,
                                                            lambda_user, lambda_item)

    return rmse_tr, rmse_te


def run_mf_cv_lambda_user(ratings, k_fold, num_epochs, num_features, lambdas_user, lambda_item):
    """ Performs cross-validation with variable lambda user 
    input: 
        ratings: the dataset
        k_fold: number of the folds
        num_epochs: number iterations of algorithm
        num_features: the dimension (K) of the subspace
        lambdas_user: array of values for the regularization parameter for the user feature matrix
        lambda_item: regularization parameter for the item feature matrix
    output:
        rmse_tr:  matrix of train errrors with shape (k_fold, len(lambdas_user))
        rmse_te:  matrix of test errors with shape (k_fold, len(lambdas_user))"""

    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(lambdas_user)))
    rmse_te = np.zeros((k_fold, len(lambdas_user)))

    gamma = 0.01

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for i, lambda_u in enumerate(lambdas_user):
            rmse_tr[k, i], rmse_te[k, i] = cross_validation(ratings, k_indices, k, num_epochs, gamma, num_features,
                                                            lambda_u, lambda_item)

    return rmse_tr, rmse_te


def run_mf_cv_lambda_item(ratings, k_fold, num_epochs, num_features, lambda_user, lambdas_item):
    """ Performs cross-validation with variable lambda item 
    input: 
        ratings: the dataset
        k_fold: number of the folds
        num_epochs: number iterations of algorithm
        num_features: the dimension (K) of the subspace
        lambda_user: regularization parameter for the user feature matrix
        lambdas_item: array of values for the regularization parameter for the item feature matrix
    output:
        rmse_tr:  matrix of train errrors with shape (k_fold, len(lambdas_item))
        rmse_te:  matrix of test errors with shape (k_fold, len(lambdas_item))"""

    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(lambdas_item)))
    rmse_te = np.zeros((k_fold, len(lambdas_item)))

    gamma = 0.01

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for i, lambda_i in enumerate(lambdas_item):
            rmse_tr[k, i], rmse_te[k, i] = cross_validation(ratings, k_indices, k, num_epochs, gamma, num_features,
                                                            lambda_user, lambda_i)

    return rmse_tr, rmse_te


############################################################################################
# Cross-validation for comparison with other models
############################################################################################
def run_mf_cv(ratings, k_fold, num_epochs, num_features):
    """cross validation for the non-regularized matrix factorization
    input: 
        ratings: the dataset
        k_fold: number of the folds
        num_epochs: number iterations of algorithm
        num_features: the dimension (K) of the subspace
    output:
        rmse_tr: train errors for every fold
        rmse_te: test errors for every fold
    """

    rmse_tr, rmse_te = run_mf_reg_cv(ratings, k_fold, num_epochs, num_features, 0, 0)
    return rmse_tr, rmse_te


def run_mf_reg_cv(ratings, k_fold, num_epochs, num_features, lambda_user, lambdas_item):
    """ Performs cross-validation for MF regularized
    input: 
        ratings: the dataset
        k_fold: number of the folds
        num_epochs: number iterations of algorithm
        num_features: the dimension (K) of the subspace
        lambda_user: regularization parameter for the user feature matrix
        lambda_item: regularization parameter for the item feature matrix
    output:
        rmse_tr: train errors for every fold
        rmse_te: test errors for every fold
    """

    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, 1))
    rmse_te = np.zeros((k_fold, 1))

    gamma = 0.01

    # K-fold cross-validation:
    for k in range(0, k_fold):
        rmse_tr[k], rmse_te[k] = cross_validation(ratings, k_indices, k, num_epochs, gamma, num_features, lambda_user,
                                                  lambdas_item)

    return rmse_tr, rmse_te


############################################################################################
# Compute the full prediction matrix once all the hyper-parameters have been chosen
############################################################################################
def mf_gd_compute_predictions(data, num_epochs, gamma, num_features, lambda_user, lambda_item):
    """ Compute the full prediction matrix 
    input: 
        data: the dataset
        k_fold: number of the folds
        num_epochs: number iterations of algorithm
        gamma: the learning rate
        num_features: the dimension (K) of the subspace
        lambda_user: regularization parameter for the user feature matrix
        lambda_item: regularization parameter for the item feature matrix
    output:
        X_hat: the predictions
        rmse: the train error on the dataset"""
    # init matrix
    Z_opt, W_opt = init_MF(data, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = data.nonzero()
    nz_data = list(zip(nz_row, nz_col))

    rmse = np.zeros((num_epochs, 1))

    print("learn the matrix factorization using GD...")
    for it in range(num_epochs):
        if (it % 5 == 0):
            print("Starting epoch number %d" % (it))
        # decrease step size
        gamma /= 1.2

        for d, n in nz_data:
            e = data[d, n] - prediction(W_opt[:, d], Z_opt[:, n])
            Z_opt[:, n] += gamma * (e * W_opt[:, d] - lambda_user * Z_opt[:, n])
            W_opt[:, d] += gamma * (e * Z_opt[:, n] - lambda_item * W_opt[:, d])


        rmse[it] = compute_error(data, Z_opt, W_opt, nz_data)

    X_hat = prediction(W_opt, Z_opt)

    return X_hat, rmse
