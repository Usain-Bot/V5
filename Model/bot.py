import tensorflow
import pandas as pd
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Data import DataPrepocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from datetime import datetime
import random
from keras.layers.recurrent import LSTM


def TrainNeuralNetwork(test_size=0.3, random_state=1):
    #load dataset
    data = DataPrepocessing.clean_data()
    dataset = data['df']
    
    #split data into X input and Y output
    X, y = dataset[:, :-1], dataset[:, -1]
    #print(dataset)
    X, y = X.astype('float'), y.astype('float')
    n_features = X.shape[1]


    #split data into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # define the keras model
    model = Sequential()
    model.add(Dense(22, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(6, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))


    # compile the keras model
    model.compile(loss='mse', optimizer='adam')

    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

    # evaluate on test set
    yhat = model.predict(X_test)

    yhat_train = model.predict(X_train)
    error = mean_absolute_error(y_test, yhat)
    print('MAE: %.3f' % error)

    scaler = data['normalizer']

    #print(y_test)
    resultat = []

    for i in range(len(y_test)):
        resultat.append([X_test[i][0], y_test[i], yhat[i][0]])
    real_result = pd.DataFrame(resultat, columns=["Prix à t", "Prix à t+1", "Estimation t+1"])

    #shuffle the real values (timestamp, open and close)
    real_values = sklearn.utils.shuffle(data['real values'], random_state=random_state)

    #Here we take first values of real_values because to test, we take first values of data too.
    try:
        #sometimes we have to take values to int(test_size*len(dataset)+1)
        real_values = real_values[:int(test_size*len(dataset)+1)].reset_index()

        #Create a dataframe to see what happened
        r = pd.DataFrame({
        'Prix à t': DataPrepocessing.invTransform(real_result['Prix à t'], scaler, 0), 
        'Prix à t+1': DataPrepocessing.invTransform(real_result['Prix à t+1'], scaler, 0),
        'Estimation à t+1': DataPrepocessing.invTransform(real_result["Estimation t+1"], scaler, 0),
        "Timestamp": real_values["Timestamp"],
        "Real open": real_values["Real open"],
        "Real close": real_values["Real close"]
        })

        
    except ValueError:
        #sometimes we have to take values to int(test_size*len(dataset)+1)
        real_values = real_values[:int(test_size*len(dataset))].reset_index()

        r = pd.DataFrame({
        'Prix à t': DataPrepocessing.invTransform(real_result['Prix à t'], scaler, 0), 
        'Prix à t+1': DataPrepocessing.invTransform(real_result['Prix à t+1'], scaler, 0),
        'Estimation à t+1': DataPrepocessing.invTransform(real_result["Estimation t+1"], scaler, 0),
        "Timestamp": real_values["Timestamp"],
        "Real open": real_values["Real open"],
        "Real close": real_values["Real close"]
        })

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

    return {'model': model, 'scaler': scaler}
