# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
    #ax2.set_xticks(np.arange(0, 2000, 300))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.01, markersize=0.05, aspect='auto')
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Training data")
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.01, markersize=0.05, aspect='auto')
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test data")
    
    plt.savefig("train_test")
    plt.show()
    
    
def rmse_visualization_vs_num_features(num_features, rmse_train, rmse_test, filename):

    #print("Minimum mean:", np.amin(mean_error_rate), ", lambda:", np.argmin(mean_error_rate))
    
    plt.plot(
        num_features,
        rmse_train,
        'r',
        linestyle="-",
        label='RMSE train')
    
    plt.plot(
        num_features,
        rmse_test,
        'b',
        linestyle="-",
        label='RMSE test')
    plt.xlabel("Number of features")
    plt.ylabel("RMSE")
    plt.legend(loc=2)
    plt.grid(True)
    plt.title("RMSE versus Number of features")
    plt.savefig(filename)
    #plt.clf() 


def rmse_visualization_vs_lambdas(lambdas_user, lambdas_item, rmse_train, rmse_test, filename):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    z1 = rmse_train.squeeze(axis=1)
    z2 = rmse_test.squeeze(axis=1)
    x = lambdas_user.squeeze(axis=1)
    y = lambdas_item.squeeze(axis=1)
    ax.plot(x, y, z1, label='RMSE train')
    ax.plot(x, y, z2, label='RMSE test')
    ax.legend()
    title = "RMSE versus lambdas"
    plt.title(title)
    plt.show()
    #plt.clf()

def rmse_visualization_vs_lambdas_user(lambdas_user, lambda_item, rmse_train, rmse_test, filename):

    plt.plot(
        lambdas_user,
        rmse_train,
        'r',
        linestyle="-",
        label='RMSE train')

    plt.plot(
        lambdas_user,
        rmse_test,
        'b',
        linestyle="-",
        label='RMSE test')
    plt.xlabel("Lambda user")
    plt.ylabel("RMSE")
    plt.legend(loc=2)
    plt.grid(True)
    title = "RMSE versus lambda user (lambda item = %s)" % (lambda_item)
    plt.title(title)
    plt.savefig(filename)
    #plt.clf()

def rmse_visualization_vs_lambdas_item(lambdas_item, lambda_user, rmse_train, rmse_test, filename):

    plt.plot(
        lambdas_item,
        rmse_train,
        'r',
        linestyle="-",
        label='RMSE train')

    plt.plot(
        lambdas_item,
        rmse_test,
        'b',
        linestyle="-",
        label='RMSE test')
    plt.xlabel("Lambda item")
    plt.ylabel("RMSE")
    plt.legend(loc=2)
    plt.grid(True)
    title = "RMSE versus lambda item (lambda user = %s)" % (lambda_user)
    plt.title(title)
    plt.savefig(filename)
    #plt.clf()