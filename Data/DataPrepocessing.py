
import APIs.Binance
import pprint
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dateutil.relativedelta import *
import numpy as np
import pytz

import time
import config as cfg


pp = pprint.PrettyPrinter(indent=3)

def regenerate_data():
    '''This function re-generate csv : data_IA.csv (file used as input of IA model)'''
    data = pd.DataFrame()
    #initialize a boolean to check if it first time in forloop (because can't concat an empty dataframe)
    first = True
    for pair in cfg.PAIRS: #we want a data with all coins pairs we choose in config.py
        data_binance = APIs.Binance.GetData(pair=pair, since=cfg.SINCE) #get data from Binance API
        if first:
            data = data_binance['df']                                               #if it is the first data : create a dataframe with this data
            first = False
            real_df = data_binance['real values']  
        else:                                                                       #else add it to the right of previous one
            data = pd.concat((data, data_binance['df']),  axis=1, join="inner")

    # To add 'Wanted' column
    wanted_result = data.iloc[:, 0].tolist() #create a new list with wanted result (which is a copy of "open" column)
    data.drop([len(data)-2, len(data)-1], inplace=True) #delete last 2 row of our data (which are the younger one)
    data['Wanted'] = wanted_result[2:] #add it "wanted" column without first two results (which are the older one) to create a lag

    #Save data in csv files
    data.to_csv(cfg.PATH_TO_STORAGE+"/data_IA.csv", index = False)
    real_df[:-2].to_csv(cfg.PATH_TO_STORAGE+"/real_values.csv", index = False)



def clean_data():
    '''This function is to prepare data to be used by IA model by normalize it and keep this normalization'''

    #Try to get data from .csv files
    try:
        data = pd.read_csv(cfg.PATH_TO_STORAGE+"/data_IA.csv")
        real_df = pd.read_csv(cfg.PATH_TO_STORAGE+"/real_values.csv")

    #if not exist generate it with regenerate_data()
    except:
        print("Missing data. We are generating it for you :)")
        regenerate_data()
        data = pd.read_csv(cfg.PATH_TO_STORAGE+"/data_IA.csv")
        real_df = pd.read_csv(cfg.PATH_TO_STORAGE+"/real_values.csv")
    
    # Normalize the data
    scaler = MinMaxScaler().fit(data)
    data = scaler.transform(data)

    return {'df': data, 'real values': real_df, 'normalizer': scaler}