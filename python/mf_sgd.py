import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

from helpers import *

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    
    # ***************************************************
    # you should return:
    #     user_features: shape = num_features, num_user
    #     item_features: shape = num_features, num_item
    # ***************************************************
    
    num_items, num_users = train.shape
    
    user_features = np.random.random((num_features,num_users))
    item_features = np.random.random((num_features,num_items))
    
    return user_features, item_features
   
def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # ***************************************************
    # calculate rmse (we only consider nonzero entries.)
    # ***************************************************
    wz = np.transpose(item_features).dot(user_features)
    
    mse = 0
    for d, n in nz:
        mse += np.power(data[d,n]-wz[d,n], 2)
        
    rmse = np.sqrt(mse) # The factor 2 disappears because of the 1/2 of MSE
    
    return mse

def prediction(W,Z):
    # W is K features x D items
    # Z is K features x N users
    return np.dot(W.T,Z)

def mf_sgd(train, test, num_epochs, gamma, num_features, lambda_user, lambda_item):
    """matrix factorization by SGD."""
    rmse_train = 0

    # set seed
    np.random.seed(988)

    # init matrix
    #user_features, item_features = init_MF(train, num_features)
    Z, W = init_MF(train, num_features)
    #print("Z:", Z.shape)
    #print("W:", W.shape)

    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            e = train[d,n] - prediction(W[:,d],Z[:,n])
            Z[:,n] += gamma * (e*W[:,d] - lambda_item*Z[:,n])
            W[:,d] += gamma * (e*Z[:,n] - lambda_user*W[:,d])

        nz_row, nz_col = train.nonzero()
        nz_train = list(zip(nz_row, nz_col))
        rmse_train = compute_error(train,Z,W, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse_train))

    # evaluate the test error.
    rmse_test = compute_error(test, Z, W, nz_test)
    print("RMSE on test data: {}.".format(rmse_test))
    
    return rmse_train, rmse_test