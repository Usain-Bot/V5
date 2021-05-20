import pandas as pd
import config as cfg
from Model import bot
import matplotlib.pyplot as plt
import numpy as np
from Data import DataPrepocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import joblib

def evaluate_model(model):
    #Try to get data from .csv files
    try:
        dataset = pd.read_csv(cfg.PATH_TO_STORAGE+"/validation_data.csv")

    #if not exist then error
    except Exception as e:
        print("Error : there is no validation_data.csv")
        raise(e)
        quit()

    #split data into X input and Y output
    timestamp, y = dataset["Timestamp"].values, dataset["Wanted"].values 
    
    #drop "timestamp" column to don't send it to model
    dataset.drop(["Timestamp"], axis=1, inplace=True)

    #load the scaler
    scaler = joblib.load(cfg.PATH_TO_STORAGE+"/scaler.save")
    #normalize with scaler
    dataset = scaler.transform(dataset)

    #separate values between "features" and "wanted" values
    X = dataset[:, :-1]
    X = X.astype('float')

    # get model predictions
    predictions = model.predict(X)

    #get the index of last column of data to be able to invtransform
    last_column = len(cfg.PAIRS)

    predictions = DataPrepocessing.invTransform(predictions, scaler, last_column)

    timestamp = [datetime.fromtimestamp(x) for x in timestamp/1000] #convert to real time

    #plot wanted and predictions to compare
    plt.plot(timestamp, y, 'b', lw=0.5)
    plt.plot(timestamp, predictions, 'r', lw=0.5)
    plt.show()


def strategy(data):

    '''First strategy is to buy when prediction is upper than now'''
    resultat = []
    wallet = cfg.WALLET
    for index, row in data.iterrows():
        if row["Estimation à t+1"] - row["Prix à t"] > 0:
            resultat.append(row["Prix à t+1"]/row["Prix à t"] * wallet - wallet)
        else:
            resultat.append(0)
        wallet += resultat[-1]
    return resultat
'''
r = bot.TrainNeuralNetwork()
argent = strategy(r)

print(sum(argent))


plt.plot(r["Timestamp"].values, np.cumsum(argent), 'r')
plt.show()
'''

evaluate_model(bot.TrainNeuralNetwork())