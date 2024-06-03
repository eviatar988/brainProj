import numpy as np

# Example matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Number of random columns to select
num_random_columns = 2

# Randomly select column indices
random_column_indices = np.random.choice(matrix.shape[1], num_random_columns, replace=False)

# Extract random columns
random_columns = matrix[:, random_column_indices]
print("Random Columns:\n", random_columns)