from exputils import *
import time
import pandas as pd
import numpy as np
import Oger
import pylab
import mdp

def DoESN(data, n_test, config):
    try:
        n_in, n_out, res_size, s_radius, leak_rate = config
        sData, ma, mi, me = scale(data)
        
        supData = series_to_supervised(sData, n_in, n_out)
        train,test = train_test_split(supData, n_test)
        train_x, train_y = [np.array(train[:, :n_in])], [np.array(train[:, n_in:])]
        test_x, test_y = [np.array(test[:, :n_in])], [np.array(test[:, n_in:])]
        reservoir = Oger.nodes.LeakyReservoirNode(input_dim=n_in, 
                                                    output_dim=res_size, 
                                                    spectral_radius=s_radius,
                                                    leak_rate=leak_rate)
        readout = Oger.nodes.RidgeRegressionNode()
        flow = mdp.Flow([reservoir, readout], verbose=1)
        data = [None, zip(train_x,train_y)]

        blockPrint()
        t0=time.time()
        flow.train(data)
        train_time = time.time() - t0
        
        t0=time.time()
        y_hat = flow(test_x)
        predi_time = time.time() - t0
        enablePrint()

        p_data = unscaleOger(y_hat, ma, mi, me)

        rmse = Oger.utils.rmse(unscaleOger(test_y[0], ma, mi, me), p_data)

        return rmse, train_time, predi_time, p_data.flatten()

    except:
        return None

def GetConfigsESN():
    n_ins = [5, 10, 20]
    n_outs = [1]
    res_sizes = [50, 100, 200, 400, 600, 1000]
    n_leaking_rate = [0.1, 0.2, 0.3, 0.5, 0.8]
    spectral_rads = [0.1, 0.3, 0.5, 0.7, 0.9]
    # create configs
    configs = list()
    for i in n_ins:
        for j in n_outs:
            for k in res_sizes:
                for l in n_leaking_rate:
                    for m in spectral_rads:
                        cfg = [i, j, k, l, m]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs, ['n_ins', 'n_outs', 'res_sizes', 'n_leaking_rate', 'spectral_rads']

configs, configNames = GetConfigsESN()
RunExp('esn', DoESN, configs, configNames)