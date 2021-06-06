#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 20:58:11 2021

@author: teiferman27
"""
#NEURAL PROHET MODEL


#TO DO :COINCIDER JEU DE DONNES TEST ET TRAIN AVEC LE BOT
#-> fichier bot.py  TrainNeuralNetwork ?? split en dehors de la fonction


def plot_forecast(model, data, periods, historic_pred=True, highlight_steps_ahead=None):
    """ plot_forecast function - generates and plots the forecasts for a NeuralProphet model
    - model -> a trained NeuralProphet model
    - data -> the dataframe used for training
    - periods -> the number of periods to forecast
    - historic_pred -> a flag indicating whether or not to plot the model's predictions on historic data
    - highlight_steps_ahead -> the number of steps ahead of the forecast line to highlight, used for autoregressive models only"""
    
    future = model.make_future_dataframe(data,periods=periods,n_historic_predictions=historic_pred)
    forecast = model.predict(future)
    
    if highlight_steps_ahead is not None:
        model = model.highlight_nth_step_ahead_of_each_forecast(highlight_steps_ahead)
        model.plot_last_forecast(forecast)
    else:    
        model.plot(forecast)



import matplotlib.pyplot as plt
import pandas as pd
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed 
from datetime import datetime

set_random_seed(0)


#load datas
data = pd.read_csv('./Data/data_IA.csv')
data_real = pd.read_csv('./Data/real_values.csv')

#see datas
data.head(10)
data_real.head(10)

#keep only Open price : pre-processing  for prophet model
data_real.drop(['Real close'], axis=1,inplace  =True)
ds, y = data_real["Timestamp"].values, data_real["Real open"].values 
plt.plot(ds,y)
Dates = [datetime.fromtimestamp(x) for x in ds/1000] #convert to real time

d_sel = {'ds': Dates,'y':y}
data_prophet = pd.DataFrame(d_sel)

#Timestamp not good for prophet model, date better
#data_prophet = data_real.rename(columns={'Timestamp': 'ds', 'Real open': 'y'}) # the usual preprocessing routine


#Modele
model = NeuralProphet(n_changepoints=100,trend_reg=0.05,yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)

metrics = model.fit(data_prophet, validate_each_epoch=True, valid_p=0.2, freq='D',plot_live_loss=True,epochs=100)

plt.figure(1)
plt.plot(metrics['SmoothL1Loss'])
plt.plot(metrics['SmoothL1Loss_val']) 



#prediction
plot_forecast(model, data_prophet, periods=7,historic_pred=False)
                                                                  
                                                                  
                                                                  
#plt.figure(figsize=(10, 7))



#Other model  : yearly_seasonality=True
model2 = NeuralProphet(n_changepoints=100,trend_reg=0.05,yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
metrics2 = model2.fit(data_prophet, validate_each_epoch=True, valid_p=0.2, freq='D',plot_live_loss=True,epochs=100)
plot_forecast(model2, data_prophet, periods=60, historic_pred=True)


#AR
model_AR = NeuralProphet(n_forecasts=60,n_lags=60,changepoints_range=0.95,n_changepoints=100,yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False,batch_size=64,epochs=100,learning_rate=1.0)
metrics_AR = model_AR.fit(data_prophet,valid_p=0.2,freq='D',epochs=100)
plot_forecast(model_AR, data_prophet, periods=60, historic_pred=True)
#Doesn't work :(
                        
