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

def visualization_num_epochs(rmse_tr, rmse_te, num_epochs, filename):

    epochs_range = np.arange(1,num_epochs+1)

    plt.plot(
        epochs_range,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.5)
    plt.plot(
        epochs_range,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.5)
    plt.title("Evolution of RMSE versus epoch number")
    plt.xlabel("Epoch number")
    plt.ylabel("RMSE")
    plt.legend(loc=3)
    plt.grid(True)
    plt.savefig(filename)
    # plt.clf()

def visualization_num_epochs_best_predictions(rmse, num_epochs, filename):

    epochs_range = np.arange(1,num_epochs+1)

    print(epochs_range.shape)

    plt.plot(
        epochs_range,
        rmse,
        'b',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='RMSE',
        linewidth=3)
    plt.title("Evolution of RMSE versus epoch number")
    plt.xlabel("Epoch number")
    plt.ylabel("RMSE")
    plt.legend(loc=1)
    plt.grid(True)
    plt.savefig(filename)
    # plt.clf()


def visualization_num_features(rmse_tr, rmse_te, num_features, filename):

    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)

    plt.plot(
        num_features,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.5)
    plt.plot(
        num_features,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.5)
    plt.plot(
        num_features,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        marker='o',
        markersize=3,
        label='mean train',
        linewidth=1)
    plt.plot(
        num_features,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        marker='v',
        markersize=3,
        label='mean test',
        linewidth=1)
    plt.title("RMSE vs Number of features")
    plt.xlabel("Number of features")
    plt.ylabel("RMSE")
    plt.legend(loc=3)
    plt.grid(True)
    plt.savefig(filename)
    # plt.clf()  # needed in case of consecutive call of this function to avoid stacking unrelated plots

def visualization_lambdas_user(rmse_tr, rmse_te, lambdas_user, filename):

    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)

    plt.plot(
        lambdas_user,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.5)
    plt.plot(
        lambdas_user,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.5)
    plt.plot(
        lambdas_user,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        marker='o',
        markersize=3,
        label='mean train',
        linewidth=1)
    plt.plot(
        lambdas_user,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        marker='o',
        markersize=3,
        label='mean test',
        linewidth=1)
    plt.title("RMSE vs Lambda User")
    plt.xlabel("Lambda User")
    plt.ylabel("RMSE")
    plt.xscale('log')
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(filename)
    # plt.clf()  # needed in case of consecutive call of this function to avoid stacking unrelated plots

def visualization_lambdas_item(rmse_tr, rmse_te, lambdas_item, filename):

    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)

    plt.plot(
        lambdas_item,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.5)
    plt.plot(
        lambdas_item,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.5)
    plt.plot(
        lambdas_item,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        marker='o',
        markersize=3,
        label='mean train',
        linewidth=1)
    plt.plot(
        lambdas_item,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        marker='o',
        markersize=3,
        label='mean test',
        linewidth=1)
    plt.title("RMSE vs Lambda Item")
    plt.xlabel("Lambda Item")
    plt.ylabel("RMSE")
    plt.xscale('log')
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(filename)
    # plt.clf()  # needed in case of consecutive call of this function to avoid stacking unrelated plots
    
def visualization_lambdas(rmse_tr, rmse_te, lambdas, filename):

    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)

    plt.plot(
        lambdas,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.5)
    plt.plot(
        lambdas,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.5)
    plt.plot(
        lambdas,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        marker='o',
        markersize=3,
        label='mean train',
        linewidth=1)
    plt.plot(
        lambdas,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        marker='o',
        markersize=3,
        label='mean test',
        linewidth=1)
    plt.title("RMSE vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("RMSE")
    #plt.xscale('log')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig(filename)
    # plt.clf()  # needed in case of consecutive call of this function to avoid stacking unrelated plots


