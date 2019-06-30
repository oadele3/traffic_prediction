from exputils import *
import time
import pandas as pd
import numpy as np
from random import random
import warnings
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

def DoSARIMA(data, n_test, config):
    # one-step sarima forecast
    def sarima_forecast(history, config):
        order, sorder, trend = config
        # define model
        blockPrint()
        model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, 
                        enforce_stationarity=False, enforce_invertibility=False)
        # fit model
        t0=time.time()
        model_fit = model.fit(disp=False)
        train_time = time.time() - t0
        
        #make one step forecast
        t0=time.time()  
        yhat = model_fit.predict(len(history), len(history))
        predi_time = time.time() - t0
        enablePrint()

        return yhat[0], train_time, predi_time

    # walk-forward validation for univariate data
    def walk_forward_predict(data, n_test, cfg):
        predictions, t_times, p_times = list(), list(), list()
        train, test = train_test_split(data, n_test)
        history = [x for x in train]
        for i in range(len(test)):
            yhat, t_time, p_time = sarima_forecast(history, cfg)
            predictions.append(yhat)
            history.append(test[i])
            t_times.append(t_time)
            p_times.append(p_time)
        train_time = (sum(t_times))/float(len(t_times))
        predi_time = (sum(p_times))/float(len(p_times))
        error = measure_rmse(test, predictions)
        return error, train_time, predi_time, predictions 

    
    try:    
        return walk_forward_predict(data, n_test, config)
    except:
        return None


# create a set of sarima configs to try
def GetConfigsSARIMA(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n','c','t','ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models


RunExp('sarima', DoSARIMA, GetConfigsSARIMA())