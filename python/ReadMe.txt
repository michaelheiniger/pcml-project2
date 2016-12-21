###########################################################################################
# PCML course - Project 2, Recommender System:
Chiara Orvati, Loïs Huguenin, Michael Heiniger

# The purpose of this file is to present the main files of the Python code written 
# and used for the project.
###########################################################################################


1. Executable files:
- run.py 		: contains the code which produces the best predictions on Kaggle

- biased_mf_sgd.ipynb	: is a Python notebook built as a framework to use and
 assess biased matrix factorisation. It uses k-fold cross-validation and outputs plots of the error rate versus a wide range of values of lambda.

2. Functions files:


- baseline_predictions.py: contains the methods to compute the baseline predictions and evaluate their performance (cross_validation)

- mf_sgd.py		: contains all functions related to (regularised) matrix factorisation (cross-validations, learning algorithms, methods to compute predictions)

- biased_mf_sgd.py	: contains all functions related to biased matrix factorisation (bias computation, cross-validations, learning algorithms, methods to compute predictions)


- helpers.py 		: contains useful functions to load, modify, split the data and submit the ratings.

- plots.py		: contains functions that output plots used to assess our models.





