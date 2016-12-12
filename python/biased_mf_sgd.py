import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mf_sgd import init_MF
import helpers as h
import plots as p


def compute_error_biased(data, prediction, nz):
    """compute the loss (RMSE) of the prediction of nonzero elements."""  
    # calculate rmse (we only consider nonzero entries.)
    mse = 0
    
    for d,n in nz:
        mse += np.power(data[d,n]- prediction[d,n],2)
    
    return np.sqrt(mse/ len(nz))

def prediction_biased(W, Z, mu, user_biases, item_biases):
    """
    mu: average of all ratings
    user_biases: deviations of user means wrt mu, shape (1,N)
    item_biases: deviations of item means wrt mu, shape (D,1)
    
    X_hat has shape (D, N)
    """
    X_hat = (W.transpose().dot(Z)) + item_biases + user_biases + mu
    if X_hat.shape == (1,1):
        return X_hat[0,0]
    return X_hat


def compute_biases(ratings):
    """computes biases of the given nonzero ratings (the training set)
    mu: overall average
    user_biases: deviations of user means wrt mu, shape (1,N)
    item_biases: deviations of item means wrt mu, shape (D,1)
    """
    num_items, num_users = ratings.shape
    nz = ratings!=0
    
    mu = ratings[nz].mean()
    user_means = ratings.sum(axis = 0)/(nz).sum(axis=0) 
    item_means = ratings.sum(axis = 1)/(nz).sum(axis=1) 
    
    user_biases = np.reshape(user_means - mu, (1, num_users))
    item_biases = np.reshape(item_means - mu, (num_items, 1))
    
    return mu, user_biases, item_biases


############################################################################################
# Learn MF using biases and regularizers
############################################################################################

def mf_sgd_biased(train, test, num_epochs, gamma, num_features, lambda_):
    """ Matrix factorization using GD 
    output:
        rmse_train: for every epoch (stored in array of shape (num_epochs, 1) )
        rmse_test: for every epoch (...)
    """

    # Init basis (W) and coeff (Z) matrices
    Z, W = init_MF(train, num_features)
    mu, user_biases, item_biases = compute_biases(train)

    # Find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    

    # Store RMSE for each epochs
    rmse_train = np.zeros((num_epochs, 1))
    rmse_test = np.zeros((num_epochs,1))

    print("Learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            e = train[d,n] - prediction_biased(W[:,d],Z[:,n], mu, user_biases[0, n], item_biases[d,0])
            user_biases[0, n] += gamma * (e - lambda_ * user_biases[0, n])
            item_biases[d, 0] += gamma * (e - lambda_ * item_biases[d, 0])
            Z[:,n] += gamma * (e *W[:,d] - lambda_ * Z[:,n])
            W[:,d] += gamma * (e *Z[:,n] - lambda_ * W[:,d])

        nz_row, nz_col = train.nonzero()
        nz_train = list(zip(nz_row, nz_col))
        X_hat = prediction_biased(W, Z , mu, user_biases, item_biases)
        rmse_train[it] = compute_error_biased(train, X_hat, nz_train)
        rmse_test[it] = compute_error_biased(test, X_hat, nz_test)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse_train[it]))
        print("RMSE on test data: {}.".format(rmse_test[it]))
    
    return rmse_train, rmse_test



############################################################################################
# Compute the full prediction matrix once all the hyper-parameters have been chosen
############################################################################################
def mf_sgd_biased_compute_predictions(data, num_epochs, gamma, num_features, lambda_):
    """ Compute the full prediction matrix """
    # init matrix
    Z_opt, W_opt = init_MF(data, num_features)
    mu, user_biases, item_biases = compute_biases(data)

    # find the non-zero ratings indices
    nz_row, nz_col = data.nonzero()
    nz_data = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        if (it % 5 == 0):
            print("Starting epoch number %d" % (it))
        # decrease step size
        gamma /= 1.2

        for d, n in nz_data:
            e = data[d,n] - prediction_biased(W_opt[:,d], Z_opt[:,n], mu, user_biases[0, n], item_biases[d])
            user_biases[0, n] += gamma * (e - lambda_ * user_biases[0, n])
            item_biases[d, 0] += gamma * (e - lambda_ * item_biases[d, 0])
            Z_opt[:, n] += gamma * (e * W_opt[:, d] - lambda_ * Z_opt[:, n])
            W_opt[:, d] += gamma * (e * Z_opt[:, n] - lambda_ * W_opt[:, d])

    X_hat = prediction_biased(W_opt, Z_opt , mu, user_biases, item_biases)
    rmse  = compute_error_biased(data, X_hat, nz_data)

    return X_hat, rmse


############################################################################################
# Cross-Validation for paramter tuning
############################################################################################


def cross_validation_biased(ratings, k_indices, k, num_epochs, gamma, num_features, lambda_):
    """ Perform one fold of K-Fold Cross-validation for Matrix Factorization with SGD """
   
    # Form the K folds
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

    # Matrix Factorization (using Stochastic Gradient Descent)
    # Returned RMSE are arrays with rmse of each epochs
    rmse_train, rmse_test = mf_sgd_biased(train_ratings, test_ratings, num_epochs, gamma, num_features, lambda_)

    # Final RMSE is the one of last epoch
    final_rmse_train = rmse_train[-1]
    final_rmse_test = rmse_test[-1]

    return final_rmse_train, final_rmse_test


def run_mf_biased_cv_num_features(ratings, k_fold, num_epochs, num_features, lambda_):
    """ Performs cross-validation with variable number of features """

    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows,_ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(num_features)))
    rmse_te = np.zeros((k_fold, len(num_features)))

    gamma = 0.01

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for i, n_features in enumerate(num_features):
            rmse_tr[k, i], rmse_te[k, i] = cross_validation_biased(ratings, k_indices, k, num_epochs, gamma, n_features, lambda_)

    return rmse_tr, rmse_te

def run_mf_biased_cv_lambda(ratings, k_fold, num_epochs, num_features, lambdas):
    """ Performs cross-validation with variable lambda """

    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(lambdas)))
    rmse_te = np.zeros((k_fold, len(lambdas)))

    gamma = 0.01

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for i, lambda_ in enumerate(lambdas):
            rmse_tr[k, i], rmse_te[k, i] = cross_validation_biased(ratings, k_indices, k, num_epochs, gamma, num_features, lambda_)

    return rmse_tr, rmse_te



############################################################################################
# Cross-validation for comparison with other models
############################################################################################
def run_mf_cv(ratings, k_fold, num_epochs, num_features):
    """cv without regularizer"""
    rmse_tr, rmse_te = run_mf_reg_cv(ratings, k_fold, num_epochs, num_features, 0)
    return rmse_tr, rmse_te

def run_mf_reg_cv(ratings, k_fold, num_epochs, num_features, lambda_):
    """ Performs cross-validation for regularized MF with bias"""

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
            rmse_tr[k], rmse_te[k] = cross_validation_biased(ratings, k_indices, k, num_epochs, gamma, num_features, lambda_)

    return rmse_tr, rmse_te