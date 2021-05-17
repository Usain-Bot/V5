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
    model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2)

    # evaluate on test set
    yhat = model.predict(X_test)

    yhat_train = model.predict(X_train)
    error = mean_absolute_error(y_test, yhat)
    print('MAE: %.3f' % error)

    scaler = data['normalizer']

    def invTransform(data, column=0, scaler=scaler):
        dummy = pd.DataFrame(np.zeros((len(data), scaler.n_features_in_)))
        dummy[column] = data
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=dummy.columns)
        return dummy[column].values

    #print(y_test)
    resultat = []

    for i in range(len(y_test)):
        #print(X_test[i][0], y_test[i], yhat[i][0])
        resultat.append([X_test[i][0], y_test[i], yhat[i][0]])
    real_result = pd.DataFrame(resultat, columns=["Prix à t", "Prix à t+1", "Estimation t+1"])

    #print(len(real_result), len(data['real values']))
    real_values = sklearn.utils.shuffle(data['real values'], random_state=random_state)
    real_values = real_values[:int(test_size*len(dataset)+1)].reset_index()

    r = pd.DataFrame({
        'Prix à t': invTransform(real_result['Prix à t'], 0), 
        'Prix à t+1': invTransform(real_result['Prix à t+1'], 0),
        'Estimation à t+1': invTransform(real_result["Estimation t+1"], 0),
        "Timestamp": real_values["Timestamp"],
        "Real open": real_values["Real open"],
        "Real close": real_values["Real close"]
        })

    print(r)

    #accuracy = model.evaluate(X, y)
    #print('Accuracy: %.2f' % (accuracy*100))

    r.sort_values("Timestamp", inplace=True)
    r["Timestamp"] = [datetime.fromtimestamp(x) for x in r["Timestamp"]/1000] #convert tot real time
    r["ecart"] = abs(r["Estimation à t+1"] - r["Prix à t+1"]) #calculate ecart
    total = r["ecart"].mean()

    #Plot the data
    r.plot("Timestamp", ["Prix à t", 'Prix à t+1', 'Estimation à t+1'])

    #Show the plot
    plt.show()
    print(total)



    def strategybool(data):
        resultat = []
        for index, row in data.iterrows():
            if bool(random.getrandbits(1)):
                resultat.append(row["Prix à t+1"]/row["Prix à t"] * 100 - 100)
            else:
                resultat.append(0)
        return resultat

    def strategy(data):
        resultat = []
        for index, row in data.iterrows():
            if row["Estimation à t+1"] - row["Prix à t"] > 100:
                resultat.append(row["Prix à t+1"]/row["Prix à t"] * 100 - 100)
            else:
                resultat.append(0)
        return resultat

    argent = strategy(r)
    argent2 = strategybool(r)

    print(sum(argent))


    plt.plot(r["Timestamp"].values, np.cumsum(argent), 'r')
    plt.plot(r["Timestamp"].values, np.cumsum(argent2), 'b')
    plt.show()

