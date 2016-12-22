###########################################################################################
# PCML course - Project 2, Recommender System:
Chiara Orvati, Loïs Huguenin, Michael Heiniger

# The purpose of this file is to present the main files of the Python code written 
# and used for the project.
###########################################################################################


1. Executable files:
- run.py 		: contains the code which produces the best predictions on Kaggle

- biased_mf_gd.ipynb	: is a Python notebook built as a framework to use and
 assess biased matrix factorisation using GD. It uses k-fold cross-validation and outputs plots of the error rate versus a wide range of values of lambda.
 
- MF_GD.ipynb : is a Python notebook built as a framework to use and
 assess (regularized) matrix factorisation using GD . It uses k-fold cross-validation and outputs plots of the error rate versus a wide range of values of lambda.

2. Functions files:

- baseline_predictions.py: contains the methods to compute the baseline predictions and evaluate their performance (cross_validation)

- mf_gd.py		: contains all functions related to (regularised) matrix factorization with GD (cross-validations, learning algorithms, methods to compute predictions)

- mf_als.py		: contains all functions related to (regularised) matrix factorization with ALS (cross-validations, learning algorithms, methods to compute predictions)

- biased_mf_gd.py	: contains all functions related to biased matrix factorization with GD (bias computation, cross-validations, learning algorithms, methods to compute predictions)


- helpers.py 		: contains useful functions to load, modify, split the data and submit the ratings.

- plots.py		: contains functions that output plots used to assess our models.

3. How to run.py

* The training dataset must be placed in data/data_train.csv along with a sample submission file in data/sampleSubmission.csv.
Then enter:
	python run.py
to run the code (it will take some time > 20min)

* The best prediction was generated on Ubuntu 14.04 x64 kernel v.3.16, python v.3.5.2 and numpy v.1.11.1


