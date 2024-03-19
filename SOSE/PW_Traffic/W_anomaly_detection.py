from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the data
data = pd.read_csv(r'./traffic.csv')

# Convert the 'DateTime' column to a datetime object
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Set the 'DateTime' column as the index
data.set_index('DateTime', inplace=True)

# Remove any missing values
data.dropna(inplace=True)

# Drop irrelevant columns
data.drop(['Junction', 'ID'], axis=1, inplace=True)

# Define the length of the input sequence
seq_length = 24

# Create empty lists to hold input/output sequences
X = []
y = []

# Iterate over the data to create sequences
for i in range(len(data)-seq_length-1):
    X.append(data.iloc[i:i+seq_length])
    y.append(data.iloc[i+seq_length])

# Convert the input/output sequences to numpy arrays
X = np.array([np.array(x) for x in X])
y = np.array([np.array(x) for x in y])

# Determine the number of samples in the dataset
num_samples = len(X)

# Split the data into training and test sets
X_train, y_train = X[:int(0.8*num_samples)], y[:int(0.8*num_samples)]
X_test, y_test = X[int(0.8*num_samples):], y[int(0.8*num_samples):]

# Shuffle the data
indices = np.arange(num_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split the data into training and test sets
X_train, y_train = X[:int(0.8*num_samples)], y[:int(0.8*num_samples)]
X_test, y_test = X[int(0.8*num_samples):], y[int(0.8*num_samples):]

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, 1)))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')


from keras.models import load_model

# Load the model
model = load_model('./model.keras')

# Train the model
# model.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test))
# model.save('model.keras')

# Evaluate the model on the test set
# mse = model.evaluate(X_test, y_test)
# print('Mean Squared Error:', mse)

# Make predictions on the test set
time = datetime.now()
y_pred = model.predict(X_test)
print(datetime.now() - time)

# Plot the actual and predicted values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()