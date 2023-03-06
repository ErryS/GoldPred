# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv('dataset.csv') # change the file name if needed
df['date'] = pd.to_datetime(df['date']) # convert date column to datetime format
df = df.set_index('date') # set date column as index
df = df.sort_index() # sort the data by date

# Scale the data to [0,1] range using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Split the data into features (X) and target (y)
X = scaled_data[:, 1:] # all columns except the first one (Daily Gold Price)
y = scaled_data[:, 0] # only the first column (Daily Gold Price)

# Define a function to create a supervised learning dataset from time series data
def create_dataset(X, y, time_steps):
    X_data = []
    y_data = []
    for i in range(len(X) - time_steps):
        X_data.append(X[i:(i + time_steps), :])
        y_data.append(y[i + time_steps])
    return np.array(X_data), np.array(y_data)

# Create a supervised learning dataset using 24 time steps
time_steps = 24 
X_data, y_data = create_dataset(X, y, time_steps)

# Define a function to split the data into train and test sets based on a given ratio
def train_test_split(X_data, y_data, ratio):
    train_size = int(len(X_data) * ratio)
    X_train = X_data[:train_size]
    y_train = y_data[:train_size]
    X_test = X_data[train_size:]
    y_test = y_data[train_size:]
    return X_train, y_train, X_test, y_test

# Define a function to create and fit an lstm model with a given number of units and epochs
def create_fit_lstm_model(X_train, y_train, units, epochs):
    model = Sequential()
    model.add(LSTM(units=units,input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train,y_train,batch_size=32,
              epochs=epochs,
              verbose=0,
              shuffle=False)
    
    return model

# Define a function to evaluate an lstm model using RMSE and MAPE metrics on test set 
def evaluate_lstm_model(model,X_test,y_test):
    
    # Make predictions on test set 
    y_pred=model.predict(X_test)
    
     # Rescale predictions and actual values back to original scale 
     scaler.scale_[0]=scaler.scale_[0]*len(df.columns) 
     scaler.scale_[1:]=scaler.scale_[1:]*len(df.columns) 
     scaler.min_[0]=scaler.min_[0]*len(df.columns) 
     scaler.min_[1:]=scaler.min_[1:]*len(df.columns) 
    
     inv_y_pred=scaler.inverse_transform(np.concatenate((y_pred,np.zeros((y_pred.shape[0],len(df.columns)-1))),axis=1))[:,0] 
     inv_y_test=scaler.inverse_transform(np.concatenate((y_test.reshape(-1 , 1),np.zeros((y_test.shape[0],len(df.columns)-1))),axis=1))[:,0] 
    
     # Calculate RMSE and MAPE metrics 
     rmse=np.sqrt(mean_squared_error(inv_y_test , inv_y_pred)) 
     mape=mean_absolute_percentage_error(inv_y_test , inv_y_pred)*100 
    
     return rmse , mape

# Define a function to plot the actual vs predicted values on test set using matplotlib library  
def plot_lstm_forecast(model,X_test,y_test): 
    
      # Make predictions on test set  
      y_pred=model.predict(X_test) 
    
      # Rescale predictions and actual values back to original scale  
      scaler.scale_[0]=scaler.scale_[0]*len(df.columns)  
      scaler.scale_[1:]=scaler.scale_[1:]*len(df.columns)  
      scaler.min_[0]=scaler.min_[0]*len(df.columns)  