import pandas as pd
import config as cfg
from Model import bot
import matplotlib.pyplot as plt
import numpy as np

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

r = bot.TrainNeuralNetwork()
argent = strategy(r)

print(sum(argent))


plt.plot(r["Timestamp"].values, np.cumsum(argent), 'r')
plt.show()
