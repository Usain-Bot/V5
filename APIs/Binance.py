from binance.client import Client
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot
import pandas as pd
from APIs import keys
import os
import pprint
from dateutil.relativedelta import *
from datetime import datetime

pp = pprint.PrettyPrinter(indent=3)

def GetData(pair, since):

    #Connect to the Binance client
    client = Client(keys.apiKey, keys.secretKey)
    # Get the data from Binance
    df = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1HOUR, since)

    # Store the time, open and close values in a pandas dataframe
    real_values = []
    for i in df:
        real_values.append([i[0], i[1], i[4]])
    real_df = pd.DataFrame(real_values, columns=["Timestamp","Real open", "Real close"])


    # Transform the data to a pandas array
    df = pd.DataFrame(df,
    columns=["Open time", "Open", "High", "Low", "Close", "Volume","Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])

    # Drop useless columns
    df.drop(["Open time", "Close time", "Ignore"], axis=1, inplace=True)

    #Plot the data
    #df.plot()

    #Show the plot
    #pyplot.show()
    
    return {'df': df, 'real values': real_df}


def convert_to_timestamp(date):
    return int(date.timestamp()*1000)