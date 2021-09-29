import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf

import os
import datetime

#import IPython
#import IPython.display
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from cleanData import x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y
#print(y[20:30])

# Step 2: Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
x1 = scaler.fit_transform(x1)
x2 = scaler.fit_transform(x2)
x3 = scaler.fit_transform(x3)
x4 = scaler.fit_transform(x4)
x5 = scaler.fit_transform(x5)
x6 = scaler.fit_transform(x6)
x7 = scaler.fit_transform(x7)
x8 = scaler.fit_transform(x8)
x9 = scaler.fit_transform(x9)
x10= scaler.fit_transform(x10)
yy = scaler.fit_transform(y)
#print(x10[:15])

# Step 3 : horizontally stack columns
stacked = hstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,yy))
print ("stacked.shape" , stacked.shape)
#print(stacked[:10])


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
    # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# choose a number of time steps #change this accordingly
# <----------- step_in, 2 year
# xxxxxxxxxxxxxxxxxxxxxx
#            ------> step_out, include current step
n_steps_in, n_steps_out = 48 , 24
# covert into input/output
X, yy = split_sequences(stacked, n_steps_in, n_steps_out)
print ("X.shape" , X.shape)
n_datasets,n_steps_in,n_features = X.shape
#print(X[:3])

print ("y.shape" , yy.shape)
n_datasets,n_steps_out = yy.shape
#print(y[:3])

# spliting data
# total 58
# 45 year : 3 year : 10 year
split_point = 540
split_point2 = 36+split_point
train_X , train_y = X[:split_point, :] , yy[:split_point, :]
valid_X , valid_y = X[split_point:split_point2, :] , yy[split_point:split_point2, :]
test_X, test_y = X[split_point2:, :], yy[split_point2:, :]

np.random.seed(42)
tf.random.set_seed(42)

#optimizer learning rate
opt = keras.optimizers.Adam(learning_rate=0.001)
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(loss='mse' , optimizer=opt , metrics=['mse'])

#model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
#    tf.keras.layers.LSTM(36, return_sequences=True,input_shape=(n_steps_in, n_features) ),
    # Shape => [batch, time, features]
#    tf.keras.layers.Dense(units=n_steps_out)
#])
#model.compile()
#model.compile(
#    optimizer='adam',
#    loss='mean_absolute_error',
#    metrics=['mean_absolute_error']
#)
#model.compile(loss='mse' , optimizer=opt , metrics=['mse'])


print(model.summary())
# Fit network
print('train_x shape:', train_X.shape)
print('train_y shape:', train_y.shape)
print('valid_x shape:', valid_X.shape)
print('valid_y shape:', valid_y.shape)
history = model.fit(train_X , train_y , epochs=45 , verbose=0 ,validation_data=(valid_X, valid_y) ,shuffle=False)

def prep_data(x_test, y_test , start , end , last):
    #prepare test data X
    #dataset_test = hstack((x1_test_scaled, x2_test_scaled))
    dataset_test_X = x_test[start:end, :]
    print("dataset_test_X :",dataset_test_X.shape)
    test_X_new = dataset_test_X.reshape(1,dataset_test_X.shape[0],dataset_test_X.shape[1])
    print("test_X_new :",test_X_new.shape)
#prepare past and groundtruth
    past_data = y_test[:end, :]
    dataset_test_y = y_test[end-1:last-1 , :]
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler1.fit(dataset_test_y)
    print("dataset_test_y :",dataset_test_y.shape)
    print("past_data :",past_data.shape)
#predictions
    y_pred = model.predict(test_X_new)
    y_pred_inv = scaler1.inverse_transform(y_pred)
    y_pred_inv = y_pred_inv.reshape(n_steps_out,1)
    y_pred_inv = y_pred_inv[:,0]
    print("y_pred :",y_pred.shape)
    print("y_pred_inv :",y_pred_inv.shape)
    
    return y_pred_inv , dataset_test_y , past_data
#start can be any point in the test data 4year, 0-24
start =20
end = start + n_steps_in 
last = end + n_steps_out 
stacked_x = hstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10))
stacked_x = stacked_x[split_point2:, :]
print('stacked_x shape: ',stacked_x.shape)
y_test = y[split_point2:, :]
print('y test shape', y_test.shape)
y_pred_inv , dataset_test_y , past_data = prep_data(stacked_x , y_test , start , end , last)

#plt.plot(y)
#plt.xlabel('Time step' ,  fontsize=18)
#plt.ylabel('y-value' , fontsize=18)
#plt.show()


print(y_pred_inv)
print(dataset_test_y)

# Plot history and future
def plot_multistep(history, prediction1 , groundtruth , start , end):
    plt.figure(figsize=(20, 4))
    #y_mean = mean(prediction1)
    range_history = len(history)
    range_future = list(range(range_history-1, range_history-1 + len(prediction1)))
    if len(groundtruth) < len(prediction1):
        for i in range(len(prediction1)-len(groundtruth)):
            groundtruth = np.append(groundtruth, groundtruth[-1])
    plt.plot(np.arange(range_history), np.array(history), label='History')
    plt.plot(range_future, np.array(prediction1),label='Forecasted with LSTM')
    plt.plot(range_future, np.array(groundtruth),label='GroundTruth')
    plt.legend(loc='upper right')
    #plt.title("Test Data from {} to {} , Mean = {:.2f}".format(start, end, y_mean) ,  fontsize=18)
    plt.xlabel('Time step')
    plt.ylabel('rates' )
    plt.title('Multivariate and Multistep LSTM Model Using Time Series to Forecast')
    plt.show()
plot_multistep(past_data , y_pred_inv , dataset_test_y , start , end)
