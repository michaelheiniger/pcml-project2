import numpy as np
import scipy.sparse as sp
from mf_sgd import init_MF
import helpers as h


def compute_error_biased(data, prediction, nz):
    """compute the loss (RMSE) of the prediction of nonzero elements.
    
    input: 
        data: the known ratings
        prediction: the predicted ratings
        nz: the indices of the nonzero values in data
    output:
        the RMSE of the prediction of the nonzero elements
    """
    mse = 0

    for d, n in nz:
        mse += np.power(data[d, n] - prediction[d, n], 2)

    return np.sqrt(mse / len(nz))


def prediction_biased(W, Z, mu, user_biases, item_biases):
    """ returns the biased prediction matrix.
    
    input :
        W: item feature matrix
        Z: user feature matrix
        mu: average of all ratings
        user_biases: deviations of user means wrt mu, shape (1,N)
        item_biases: deviations of item means wrt mu, shape (D,1) 
    output:
        X_hat : has shape (D, N)
    """
    # compute the biased prediction matrix
    X_hat = (W.transpose().dot(Z)) + item_biases + user_biases + mu

    # make sure that ratings stay within valid range of 1-5
    if np.isscalar(X_hat):
        if X_hat < 1:
            X_hat = 1
        elif X_hat > 5:
            X_hat = 5
    else:
        X_hat[X_hat < 1] = 1.
        X_hat[X_hat > 5] = 5.

    return X_hat


def compute_biases(ratings):
    """computes biases for every user and every item, and the overall average of the ratings
    
    input:
        ratings: the matrix of ratings
    output: 
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu, shape (1,N)
        item_biases: deviations of item means wrt mu, shape (D,1)
    """
    num_items, num_users = ratings.shape

    # boolean array showing the nonzero entries
    nz = ratings != 0

    # mean over all nonzero ratings
    mu = ratings[nz].mean()

    # biases for every user/item
    user_means = ratings.sum(axis=0) / (nz).sum(axis=0)
    item_means = ratings.sum(axis=1) / (nz).sum(axis=1)
    user_biases = np.reshape(user_means - mu, (1, num_users))
    item_biases = np.reshape(item_means - mu, (num_items, 1))

    return mu, user_biases, item_biases


def compute_biases_restricted(ratings, num_items_per_user, num_users_per_item, min_num_ratings):
    """computes biases of the users and the items which have more than 'min_num_ratings' non-zero ratings.
    
    input:
        ratings: the matrix of ratings
        num_items_per_user: array containing the number of items for every user
        num_users_per_item: array containing the number of users for every item
        min_num_ratings: the minimum number of ratings wanted to compute the biases
    output:
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu, shape (1,N)
        item_biases: deviations of item means wrt mu, shape (D,1)
    """

    num_items, num_users = ratings.shape

    # boolean array equal to 1 when the user/item has not enough ratings
    invalid_users = np.reshape(num_items_per_user < min_num_ratings, (1, num_users))
    invalid_items = np.reshape(num_users_per_item < min_num_ratings, (num_items, 1))

    # boolean array showing the nonzero entries
    nz = ratings != 0

    # the mean is computed over ALL nonzero ratings
    mu = ratings[nz].mean()

    # compute biases item-wise/user_wise
    user_means = ratings.sum(axis=0) / (nz).sum(axis=0)
    item_means = ratings.sum(axis=1) / (nz).sum(axis=1)
    user_biases = np.reshape(user_means - mu, (1, num_users))
    item_biases = np.reshape(item_means - mu, (num_items, 1))

    # set the biases of the invalid users/items to zero
    user_biases[invalid_users] = 0
    item_biases[invalid_items] = 0

    return mu, user_biases, item_biases


############################################################################################
# Learn MF using biases and regularizers
############################################################################################

def mf_sgd_biased(train, test, num_epochs, gamma, num_features, lambda_, mu, user_biases, item_biases, step_decrease):
    """ Biased Matrix factorization using SGD
    
    input:
        train: the train data
        test: the test data
        num_epochs: the number of iterations for the algorithm
        gamma : initial step size
        num_features: dimension of the subspace (K)
        lambda_: regularization parameter
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu
        item_biases: deviations of item means wrt mu
        step_decrease: whether to decrease gamma at every iteration (True / False)
        
    output:
        rmse_train: for every epoch (stored in array of shape (num_epochs, 1) )
        rmse_test: for every epoch (...)
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

    print("Learn the matrix factorization using SGD...")
    for it in range(num_epochs):

        if step_decrease:
            # decrease step size
            gamma /= 1.2

        for d, n in nz_train:
            e = train[d, n] - prediction_biased(W[:, d], Z[:, n], mu, user_biases[0, n], item_biases[d, 0])
            user_biases[0, n] += gamma * (e - lambda_ * user_biases[0, n])
            item_biases[d, 0] += gamma * (e - lambda_ * item_biases[d, 0])
            Z[:, n] += gamma * (e * W[:, d] - lambda_ * Z[:, n])
            W[:, d] += gamma * (e * Z[:, n] - lambda_ * W[:, d])

        nz_row, nz_col = train.nonzero()
        nz_train = list(zip(nz_row, nz_col))
        X_hat = prediction_biased(W, Z, mu, user_biases, item_biases)
        rmse_train[it] = compute_error_biased(train, X_hat, nz_train)
        rmse_test[it] = compute_error_biased(test, X_hat, nz_test)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse_train[it]))
        print("RMSE on test data: {}.".format(rmse_test[it]))

    return rmse_train, rmse_test


############################################################################################
# Compute the full prediction matrix once all the hyper-parameters have been chosen
############################################################################################
def mf_sgd_biased_compute_predictions(ratings, num_epochs, gamma, num_features, lambda_, mu, user_biases, item_biases,
                                      step_decrease):
    """ Compute the full prediction matrix for the biased version of the MF 
    
    input:
        ratings: the matrix of ratings
        num_epochs: the number of iterations for the algorithm
        gamma : initial step size
        num_features: dimension of the subspace (K)
        lambda_: regularization parameter
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu
        item_biases: deviations of item means wrt mu
        step_decrease: whether to decrease gamma at every iteration (True/ False)
        
    output: 
        X_hat: the prediction matrix
        rmse : the train error of the last epch
        """
    # init matrix
    Z_opt, W_opt = init_MF(ratings, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = ratings.nonzero()
    nz_ratings = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        if (it % 5 == 0):
            print("Starting epoch number %d" % (it))

        if step_decrease:
            # decrease step size
            gamma /= 1.2

        for d, n in nz_ratings:
            e = ratings[d, n] - prediction_biased(W_opt[:, d], Z_opt[:, n], mu, user_biases[0, n], item_biases[d, 0])
            user_biases[0, n] += gamma * (e - lambda_ * user_biases[0, n])
            item_biases[d, 0] += gamma * (e - lambda_ * item_biases[d, 0])
            Z_opt[:, n] += gamma * (e * W_opt[:, d] - lambda_ * Z_opt[:, n])
            W_opt[:, d] += gamma * (e * Z_opt[:, n] - lambda_ * W_opt[:, d])

    X_hat = prediction_biased(W_opt, Z_opt, mu, user_biases, item_biases)
    rmse = compute_error_biased(ratings, X_hat, nz_ratings)

    return X_hat, rmse


############################################################################################
# Cross-Validation for parameter tuning
############################################################################################

def cross_validation_biased(ratings, k_indices, k, num_epochs, gamma, num_features, lambda_, mu, user_biases,
                            item_biases, step_decrease):
    """ Perform one fold of K-Fold Cross-validation for biased Matrix Factorization with SGD 
    
    input:
        ratings: the matrix of ratings
        k_indices: indices for cross-validation
        k: number of the fold
        num_epochs: the number of iterations for the algorithm
        gamma : initial step size
        num_features: dimension of the subspace (K)
        lambda_: regularization parameter
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu
        item_biases: deviations of item means wrt mu
        step_decrease: whether to decrease gamma at every iteration (True/False)
    output: 
        final_rmse_train: train rmse of the last epoch
        final_rmse_test: test rmse of the last epoch
    """

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
    rmse_train, rmse_test = mf_sgd_biased(train_ratings, test_ratings, num_epochs, gamma, num_features, lambda_, mu,
                                          user_biases, item_biases, step_decrease)

    # Final RMSE is the one of last epoch
    final_rmse_train = rmse_train[-1]
    final_rmse_test = rmse_test[-1]

    return final_rmse_train, final_rmse_test


def run_mf_biased_cv_num_features_lambdas(ratings, k_fold, num_epochs, gamma, num_features, lambdas, mu, user_biases,
                                          item_biases, step_decrease):
    """ Performs cross-validation with variable number of features and different values of lambda
    
    input:
        ratings: the matrix of ratings
        k_fold: number of folds
        num_epochs: the number of iterations for the algorithm
        gamma: step size
        num_features: array of different values for the dimension (K) of the subspace
        lambdas: array of different values for the regularization parameter
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu
        item_biases: deviations of item means wrt mu
        step_decrease: whether to decrease gamma at every iteration (True/False)

    output: 
        rmse_tr: matrix of train errors with shape (k_fold, len(num_features), len(lambdas)) 
        rmse_te: matrix of test errors with shape (k_fold, len(num_features), len(lambdas)) 
    """

    h.check_kfold(k_fold)

    seed = 3

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(num_features), len(lambdas)))
    rmse_te = np.zeros((k_fold, len(num_features), len(lambdas)))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for i, n_features in enumerate(num_features):
            for j, lambda_ in enumerate(lambdas):
                rmse_tr[k, i, j], rmse_te[k, i, j] = cross_validation_biased(ratings, k_indices, k, num_epochs, gamma,
                                                                             n_features, lambda_, mu, user_biases,
                                                                             item_biases, step_decrease)

    return rmse_tr, rmse_te


def run_mf_biased_cv_num_features(ratings, k_fold, num_epochs, gamma, num_features, lambda_, mu, user_biases,
                                  item_biases, step_decrease):
    """ Performs cross-validation with variable number of features 
    
    input:
        ratings: the matrix of ratings
        k_fold: number of folds
        num_epochs: the number of iterations for the algorithm
        gamma: step size
        num_features: array of different values for the dimension (K) of the subspace
        lambda_: the regularization parameter
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu
        item_biases: deviations of item means wrt mu
        step_decrease: whether to decrease gamma at every iteration (True/False)

    output: 
        rmse_tr: matrix of train errors with shape (k_fold, len(num_features)) 
        rmse_te: matrix of test errors with shape (k_fold, len(num_features)) """

    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(num_features)))
    rmse_te = np.zeros((k_fold, len(num_features)))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for i, n_features in enumerate(num_features):
            rmse_tr[k, i], rmse_te[k, i] = cross_validation_biased(ratings, k_indices, k, num_epochs, gamma, n_features,
                                                                   lambda_, mu, user_biases, item_biases, step_decrease)

    return rmse_tr, rmse_te


def run_mf_biased_cv_lambda(ratings, k_fold, num_epochs, gamma, num_features, lambdas, mu, user_biases, item_biases,
                            step_decrease):
    """ Performs cross-validation with variable lambda 
    
    input:
        ratings: the matrix of ratings
        k_fold: number of folds
        num_epochs: the number of iterations for the algorithm
        gamma: step size
        num_features: the dimension (K) of the subspace
        lambdas: array of different values for the regularization parameter
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu
        item_biases: deviations of item means wrt mu
        step_decrease: whether to decrease gamma at every iteration (True/False)

    output: 
        rmse_tr: matrix of train errors with shape (k_fold, len(lambdas)) 
        rmse_te: matrix of test errors with shape (k_fold, len(lambdas))"""

    h.check_kfold(k_fold)

    seed = 1

    # Get k folds of indices for cross-validation
    rows, _ = ratings.nonzero()
    k_indices = h.build_k_indices(len(rows), k_fold, seed)

    # Save training/test RMSE for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(lambdas)))
    rmse_te = np.zeros((k_fold, len(lambdas)))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for i, lambda_ in enumerate(lambdas):
            rmse_tr[k, i], rmse_te[k, i] = cross_validation_biased(ratings, k_indices, k, num_epochs, gamma,
                                                                   num_features, lambda_, mu, user_biases, item_biases,
                                                                   step_decrease)

    return rmse_tr, rmse_te


############################################################################################
# Cross-validation for comparison with other models (fixed parameters)
############################################################################################
def run_mf_bias_cv(ratings, k_fold, num_epochs, gamma, num_features, lambda_, mu, user_biases, item_biases,
                   step_decrease):
    """ Performs k_fold cross-validation for regularized MF with bias
    
    input:
        ratings: the matrix of ratings
        k_fold: number of folds
        num_epochs: the number of iterations for the algorithm
        gamma: step size
        num_features: the dimension (K) of the subspace
        lambda_: the regularization parameter
        mu: overall average of ratings
        user_biases: deviations of user means wrt mu
        item_biases: deviations of item means wrt mu
        step_decrease: whether to decrease gamma at every iteration (True /False)
    output:
        rmse_tr: array containing the train rmse for each fold
        rmse_te: array containing the test rmse for each fold
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
        rmse_tr[k], rmse_te[k] = cross_validation_biased(ratings, k_indices, k, num_epochs, gamma, num_features,
                                                         lambda_, mu, user_biases, item_biases, step_decrease)

    return rmse_tr, rmse_te
