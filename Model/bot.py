import tensorflow
import pandas as pd
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Data import DataPrepocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from datetime import datetime
import random
import joblib
import config as cfg



def TrainNeuralNetwork(test_size=0.1, random_state=1, steps=cfg.STEPS):
    #load dataset
    data = DataPrepocessing.clean_data()

    dataset = data['df']

    #split data into X input and Y output
    X, y = dataset[:, :-1], dataset[:, -1]
    #print(dataset)
    X = DataPrepocessing.lstm_prepare(X[::-1], steps)

    #delete last values of y because last values are not in X
    y = y[:-steps]
    y = y[::-1]

    X, y = X.astype('float'), y.astype('float')
    n_features = X.shape[2]

    # define the keras model
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(steps, n_features), dropout=0.2))
    model.add(LSTM(units=64, return_sequences=True, dropout=0.2))
    model.add(LSTM(units=32, return_sequences=True, dropout=0.2))
    model.add(LSTM(units=16, return_sequences=True, dropout=0.2))
    model.add(LSTM(units=8, dropout=0.2))

    model.add(Dense(units=1))

    # compile the keras model
    model.compile(loss='mse', optimizer='adam')

    # fit the keras model on the dataset
    model.fit(X, y, epochs=30, batch_size=100, verbose=1, shuffle=True)

    # evaluate on test set
    yhat = model.predict(X)

    error = mean_absolute_error(y, yhat)
    print('MAE: %.3f' % error)

    #Create a dataframe with predicted values and wanted values at t+1
    resultat = []

    for i in range(len(y)):
        resultat.append([X[i][0][0], y[i], yhat[i][0]])
    real_result = pd.DataFrame(resultat, columns=["Prix à t", "Prix à t+1", "Estimation t+1"])

    #Cut some values because for lstm last values are not take
    real_values = data['real values'][::-1]
    real_values = real_values[:-steps]

    #load scaler
    scaler = joblib.load(cfg.PATH_TO_STORAGE+"/scaler.save") 

    #get the index of last column of data to be able to invtransform
    last_column = len(cfg.PAIRS)

    #Create a dataframe to see what happened
    r = pd.DataFrame({
        'Prix à t': DataPrepocessing.invTransform(real_result['Prix à t'], scaler, 0), 
        'Prix à t+1': DataPrepocessing.invTransform(real_result['Prix à t+1'], scaler, last_column*9), 
        'Estimation à t+1': DataPrepocessing.invTransform(real_result["Estimation t+1"], scaler, last_column*9), 
        "Timestamp": real_values["Timestamp"],
        "Real open": real_values["Real open"],
        "Real close": real_values["Real close"]
    })

    print(r)

    #calculate model accuracy
    accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    r.sort_values("Timestamp", inplace=True) #sort values by timestamp
    r["Timestamp"] = [datetime.fromtimestamp(x) for x in r["Timestamp"]/1000] #convert to real time
    r["ecart"] = abs(r["Estimation à t+1"] - r["Prix à t+1"]) #calculate ecart
    total = r["ecart"].mean()

    #Plot the data
    r.plot("Timestamp", ['Prix à t+1', 'Estimation à t+1'], lw=0.5)

    #Show the plot
    plt.show()
    #print(total)

    return model
