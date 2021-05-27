
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
import json
import joblib

pp = pprint.PrettyPrinter(indent=3)

def regenerate_data():
    '''This function re-generate .csv files: data_IA.csv, real_values.csv and validation_data.csv (file used as input of IA model, and strategies)'''
    data = pd.DataFrame()
    #initialize a boolean to check if it first time in forloop (because can't concat an empty dataframe)
    first = True
    for pair in cfg.PAIRS: #we want a data with all coins pairs we choose in config.py
        data_binance = APIs.Binance.GetData(pair=pair, since=cfg.SINCE) #get data from Binance API
        if first:           #if it is the first data : create a dataframe with this data
            data = data_binance['df']  
            first = False
            real_df = data_binance['real values']  
        else:               #else add it to the right of previous one
            if len(data)!=len(data_binance['df']):
                #If the data is empty because Binance doesn't have values from enough time stop program (quit())
                print("Binance have not enough data for " + pair)
                print("Please change the config.py file")
                quit()                
            data = pd.concat((data, data_binance['df']),  axis=1, join="inner")

    # To add 'Wanted' column
    wanted_result = data.iloc[:, 0].tolist() #create a new list with wanted result (which is a copy of "open" column)
    data.drop([len(data)-2, len(data)-1], inplace=True) #delete last 2 row of our data (which are the younger one)
    data['Wanted'] = wanted_result[2:] #add it "wanted" column without first two results (which are the older one) to create a lag

    #Create a scaler
    scaler = MinMaxScaler().fit(data)

    #Save the scaler to use it later
    joblib.dump(scaler, cfg.PATH_TO_STORAGE+"/scaler.save") 
    
    #Save data in csv files
    '''Here we split data in two : first part is going to train/test, 2nd part is going to validation
    First part time : from the beginning to HOURS_TO_TEST ago
    2nd part time : from HOURS_TO_TEST ago to now'''

    #train/test
    data[:-cfg.HOURS_TO_TEST].to_csv(cfg.PATH_TO_STORAGE+"/data_IA.csv", index = False)
    real_df[:-cfg.HOURS_TO_TEST-2].to_csv(cfg.PATH_TO_STORAGE+"/real_values.csv", index = False)

    #validation
    validation_data = pd.concat((real_df[-cfg.HOURS_TO_TEST-1:]["Timestamp"], data[-cfg.HOURS_TO_TEST+1:]), axis=1, join="inner")
    validation_data.to_csv(cfg.PATH_TO_STORAGE+"/validation_data.csv", index = False)

def clean_data():
    '''This function is to prepare data to be used by IA model by normalize it and keep this normalization'''

    #Try to get data from .csv files
    try:
        data = pd.read_csv(cfg.PATH_TO_STORAGE+"/data_IA.csv")
        real_df = pd.read_csv(cfg.PATH_TO_STORAGE+"/real_values.csv")
        pd.read_csv(cfg.PATH_TO_STORAGE+"/validation_data.csv")

    #if not exist generate it with regenerate_data()
    except:
        print("Missing data. We are generating it for you :)")
        regenerate_data()
        data = pd.read_csv(cfg.PATH_TO_STORAGE+"/data_IA.csv")
        real_df = pd.read_csv(cfg.PATH_TO_STORAGE+"/real_values.csv")

    #Normalize data
    scaler = joblib.load(cfg.PATH_TO_STORAGE+"/scaler.save") 
    data = scaler.transform(data)

    return {'df': data, 'real values': real_df}


def invTransform(data, scaler, column=0):
    
    '''This function allows to inverse normalization of a column.
    scaler.inverse_transform function needs to have same size of array when normalization was done
    so we add 0 except on the column we want
    stackoverflow subject : https://stackoverflow.com/questions/53049396/sklearn-inverse-transform-return-only-one-column-when-fit-to-many'''

    dummy = pd.DataFrame(np.zeros((len(data), scaler.n_features_in_)))
    dummy[column] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=dummy.columns)
    return dummy[column].values



def lstm_prepare(data, steps):
    '''This function create an array with (samples, timesteps, features) 3D format
    Careful earlier features need to be first rows !!!!! '''
    result = []
    for i in range(len(data)-steps):
        result.append(data[i:i+steps])
    result = np.array(result)
    print(result.shape)
    return result
