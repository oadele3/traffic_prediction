from exputils import *
import time
import pandas as pd
import numpy as np
from random import random
import warnings
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def DoETS(data, n_test, config):
    # one-step Holt Winters Exponential Smoothing forecast
    def exp_smoothing_forecast(history, config):
        t,d,s,p,b,r = config
        # define model
        history = np.array(history)
        blockPrint()
        model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
        # fit model
        t0=time.time()
        model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
        train_time = time.time() - t0

        # make one step forecast
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
            yhat, t_time, p_time = exp_smoothing_forecast(history, cfg)
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
def GetConfigsETS(seasonal=[None]):
    models = list()
    # define config lists
    t_params = [None, 'mul','add' ]
    d_params = [False, True]
    s_params = [None, 'mul', 'add']
    p_params = seasonal
    b_params = [True, False]
    r_params = [False, True]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t,d,s,p,b,r]
                            models.append(cfg)
    return models


RunExp('ets', DoETS, GetConfigsETS(seasonal=[0,6,12]))