# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load data from csv file
data = pd.read_csv('datasetgold.csv')

# Define input and output columns
input_cols = data.columns[:-1]
output_col = data.columns[-1]

# Convert data to numpy arrays
X = data[input_cols].values
y = data[output_col].values

# Normalize input data using min-max scaling
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

# Define hyperparameters
n_splits = 5 # number of folds for cross-validation
n_features = len(input_cols) # number of input features
n_steps = 3 # number of time steps for lstm input
n_layers = 2 # number of lstm layers
n_units = 64 # number of hidden units per layer
activation = 'relu' # activation function for lstm layers
learning_rate = 0.01 # learning rate for optimizer

# Define a function to reshape data into lstm format (samples, time steps, features)
def reshape_data(data, n_steps):
    n_samples = len(data) - n_steps + 1 
    output = np.zeros((n_samples, n_steps, data.shape[1]))
    for i in range(n_samples):
        output[i] = data[i:i+n_steps]
    return output

# Reshape input data into lstm format 
X_norm_reshaped = reshape_data(X_norm, n_steps)

# Initialize lists to store results 
rmse_list = [] # list to store root mean squared error for each fold 
mape_list = [] # list to store mean absolute percentage error for each fold 

# Initialize k-fold cross-validation 
kf = KFold(n_splits=n_splits)

# Loop over each fold 
for train_index, test_index in kf.split(X_norm_reshaped):
    # Split data into training and testing sets 
    X_train, X_test = X_norm_reshaped[train_index], X_norm_reshaped[test_index]
    y_train, y_test = y[train_index + n_steps - 1], y[test_index + n_steps - 1]

    # Build lstm model 
    model = Sequential()
    model.add(LSTM(n_units, activation=activation, return_sequences=True,
                   input_shape=(n_steps,n_features)))
    for i in range(n_layers-2):
        model.add(LSTM(n_units, activation=activation, return_sequences=True))
    model.add(LSTM(n_units, activation=activation))
    model.add(Dense(1))

    # Compile model with mse loss and adam optimizer 
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # Fit model on training data 
    model.fit(X_train,y_train,batch_size=32,
              epochs=10,
              validation_data=(X_test,y_test),
              verbose=0)

    # Predict on testing data 
    y_pred=model.predict(X_test)

    # Calculate rmse and mape and append to lists 
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    
rmse_list.append(rmse)
mape=mean_absolute_percentage_error(y_test,y_pred)
mape_list.append(mape)

# Print average rmse and mape across all folds 
print('Average RMSE:',np.mean(rmse_list))
print('Average MAPE:',np.mean(mape_list))