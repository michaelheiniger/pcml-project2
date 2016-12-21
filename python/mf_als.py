import numpy as np
import scipy
import scipy.sparse as sp
import time
import helpers as h
import plots as p

def get_error(X, W, Z):
    predic = W.dot(Z.T)
    predic[predic < 1] = 1.
    predic[predic > 5] = 5.
    ctrain = sp.coo_matrix(X)
    error = 0
    n = 0
    for i,j,v in zip(ctrain.row, ctrain.col, ctrain.data):
        error += (v - predic[i, j]) ** 2
        n += 1
    return error/n

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    train_c = sp.csr_matrix(train)
    W = 5 * np.random.rand(train_c.shape[0], num_features) # Film_features
    col_avg = train.sum(axis=1).A1 / np.diff(train_c.indptr)
    W[:,0] = col_avg
    Z = 5 * np.random.rand(train.shape[1], num_features)  # User_features

    return sp.csr_matrix(W), sp.csr_matrix(Z)


def update_user_feature(
        train, item_features, lambda_user,
        user_features, num_features, A_arr):
    """update user feature matrix."""
    train_c = sp.csr_matrix(train.T)
    for u in range(train.shape[1]):
        d = A_arr[u]
        A_i = sp.diags(d).tocsr()
        #print(A_i.shape, train.shape)
        user_features[u] = sp.linalg.spsolve(item_features.T.dot(A_i.dot(item_features)) +  lambda_user * 
                                             sp.eye(num_features), item_features.T.dot(A_i.dot(train_c[u].T)))

def update_item_feature(
        train, user_features, lambda_item,
        item_features, num_features, A_arr):
    """update item feature matrix."""
    train_c = sp.csr_matrix(train)
    for m in range(train.shape[0]):
        d = A_arr[m]
        A_i = sp.diags(d).tocsr()

        item_features[m] = sp.linalg.spsolve(user_features.T.dot(A_i.dot(user_features)) +  lambda_item * 
                                             sp.eye(num_features), user_features.T.dot(A_i.dot(train_c[m].T)))


def ALS_pred(data, lambda_user, lambda_item, num_features):
    A = sp.csr_matrix(data)
    A[A > 0] = 1
    A_arr = A.toarray()

    # init ALS
    item_features, user_features = init_MF(data, num_features)
    for i in range(10):
        start = time.clock()
        print("Iteration ", i)
        update_user_feature(data, item_features, lambda_user, user_features, num_features, A_arr.T)
        update_item_feature(data, user_features, lambda_item, item_features, num_features, A_arr)
        stop = time.clock()
        print("One iteration in: ", stop-start, "s")
    return item_features, user_features

def cross_validation(ratings, k_indices, k, num_features, lambda_user, lambda_item):
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

    it, us = ALS_pred(train_ratings, lambda_user, lambda_item, num_features)
    final_rmse_test = np.sqrt(2* get_error(test_ratings, it, us))
    final_rmse_train = np.sqrt(2* get_error(train_ratings, it, us))

    return final_rmse_train, final_rmse_test

############################################################################################
# Cross-validation for comparison with other models
############################################################################################

def run_mf_reg_cv(ratings, k_fold, num_features, lambda_user, lambdas_item):
    """ Performs cross-validation for MF regularized"""

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
            rmse_tr[k], rmse_te[k] = cross_validation(ratings, k_indices, k, num_features, lambda_user, lambdas_item)

    return rmse_tr, rmse_te