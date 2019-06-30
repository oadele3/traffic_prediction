from exputils import *
import time
import pandas as pd
import numpy as np
from random import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def DoLSTM(data, n_test, config):
    
    def model_fit(train, config):
        # unpack config
        n_input, n_output, n_nodes, n_epochs, n_batch, n_diff = config
        # prepare data
        if n_diff > 0:
            train = difference(train, n_diff)
        # transform series into supervised format
        sdata = series_to_supervised(train, n_in=n_input)
        # separate inputs and outputs
        train_x, train_y = sdata[:, :-1], sdata[:, -1]
        # reshape input data into [samples, timesteps, features]
        n_features = 1
        train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
        # define model
        blockPrint()
        model = Sequential()
        model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(1))
        # fit model
        t0=time.time()
        model.compile(loss='mse', optimizer='adam')
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
        train_time = time.time() - t0
        enablePrint()
        return model, train_time

    def model_predict(model, history, config):
        # unpack config
        n_input, _, _,  _, _, n_diff = config
        # prepare data
        correction = 0.0
        if n_diff > 0:
            correction = history[-n_diff]
            history = difference(history, n_diff)
        x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
        # forecast
        yhat = model.predict(x_input, verbose=0)
        return correction + yhat[0]
    
    def walk_forward_predict(data, n_test, cfg):
        predictions = list()
        train, test = train_test_split(data, n_test)
        model, train_time = model_fit(train, cfg)
        history = [x for x in train]
        t0=time.time()
        for i in range(len(test)):
            yhat = model_predict(model, history, cfg)
            predictions.append(yhat)
            history.append(test[i])
        predi_time = time.time() - t0
        error = measure_rmse(test, predictions)
        pred_vals = [i[0]for i in predictions]
        return error, train_time, predi_time, pred_vals 

    
    try:    
        return walk_forward_predict(data, n_test, config)
    except:
        return None

def GetConfigsLSTM():
    configs = [
    [10, 1, 100, 10, 50, 96],
    [10, 1, 50, 10, 30, 96],
    [10, 1, 100, 10, 20, 96]
    ]
    # define scope of configs
    n_input = [5, 10, 20]
    n_output = [1]
    n_nodes = [50, 100]
    n_epochs = [10, 20, 50]
    n_batch = [10, 20, 30, 50]
    n_diff = [96]
    # create configs
    configs = list()
    for i in n_input:
        for ab in n_output:
            for j in n_nodes:
                for k in n_epochs:
                    for l in n_batch:
                        for m in n_diff:
                            cfg = [i, ab, j, k, l, m]
                            configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs

RunExp('lstm', DoLSTM, GetConfigsLSTM())