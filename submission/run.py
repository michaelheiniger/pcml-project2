import datetime
from src.helpers import *
from collections import deque
from src.biased_mf_gd import mf_gd_biased_compute_predictions, compute_biases

# Load dataset
path_dataset = "data/data_train.csv"
ratings = load_data(path_dataset)

# Compute initial biases
mu, user_biases, item_biases = compute_biases(ratings)

# Compute prediction matrix and associated RMSE
X_hat, rmse = mf_gd_biased_compute_predictions(ratings, 31, 0.005, 60, 0.06, mu, user_biases, item_biases, False)

# Output predictions
now = datetime.datetime.now()
now_str = now.strftime("%d-%m-%Y_%Hh%M_%S")
output_path = 'data/predictions-%s-group-clm.csv' % now_str

indices = extract_indices('data/sampleSubmission.csv')

ratings_to_write = deque()
for row, col in indices:
    ratings_to_write.append((row, col, X_hat[row,col]))
            
create_csv_submission(ratings_to_write, output_path)
