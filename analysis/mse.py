import numpy as np

# Assuming you have a list of columns with numbers
columns_with_numbers = ["$W$", "$C_1$", "$C_2$", "$C_3$"]

# Assuming you have two dictionaries with inferred and evaluated values for each column
inferred_values = {
    "$W$": [0.99, 1.00, 0.70, 0.31, 0.32, 0.00, 1.00, 0.95],
    "$C_1$": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "$C_2$": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "$C_3$": [0.82, 0.36, 0.39]
}

evaluated_values = {
    "$W$": [1.00, 0.98, 0.75, 0.28, 0.29, 0.02, 0.97, 0.99],
    "$C_1$": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "$C_2$": [0.99, 1.00, 1.00, 0.99, 1.00, 0.99],
    "$C_3$": [0.76, 0.28, 0.29]
}

# Calculate MSE for each column with numbers
for column in columns_with_numbers:
    avg = np.abs(np.subtract(inferred_values[column], evaluated_values[column])).mean()
    mse = np.square(np.subtract(inferred_values[column], evaluated_values[column])).mean()
    print(f'AVG for {column}: {avg}')
    print(f'MSE for {column}: {mse}')
