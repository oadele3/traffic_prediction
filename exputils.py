import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
import math
import csv
import time
from sklearn.metrics import mean_squared_error

def series_to_supervised(data, n_in, n_out=1):
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values

def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

def scale(data):
    maxi = np.amax(data)
    mini = np.amin(data)
    mean = float(sum(data))/len(data)
    data = [(float(i) - mean) / (maxi - mini) for i in data]
    return  data, maxi, mini, mean

def unscale(data, maxi, mini, mean):
    return [((maxi-mini)*i + mean) for i in data]

def unscaleOger(dat, ma, mi, me):
    return (ma-mi)*dat + me

def LoadData(filename, ts_len, colname, xisdate=False):
    df = pd.read_csv(filename)
    df[[colname]] = df[[colname]].fillna(method='bfill')
    print('dataset_len = ', len(df))
    if xisdate:
        df['date'] = pd.to_datetime(df['date'])
        df_idx = df.set_index(['date'], drop=True)
        df_idx = df_idx.sort_index(axis=1, ascending=True)
    data = df[[colname]][:ts_len]
    data = data.values.flatten().tolist()
    return data

def measure_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

def difference(data, order):
    return [data[i] - data[i - order] for i in range(order, len(data))]

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def StartPrintToFile(filename):
    sys.stdout = open(filename, 'w')

def ContinuePrintToFile(filename):
    sys.stdout = open(filename, 'a')

def PlotAll(filename, data, best_p_values, n_test, n_in):
    t_len = len(data)
    plt.figure(figsize=(18, 6))
    plt.plot(range(n_in, t_len-n_test), data[n_in:t_len-n_test], 'b')
    plt.plot(range(t_len-n_test, t_len), data[-n_test:], 'b')
    plt.plot(range(t_len-n_test, t_len), best_p_values, 'r')
    plt.savefig(filename)

def PlotTest(filename, data, best_p_values, n_test):
    t_len = len(data)
    plt.figure(figsize=(18, 6))
    plt.plot(range(t_len-n_test, t_len), data[-n_test:], 'b')
    plt.plot(range(t_len-n_test, t_len), best_p_values, 'r')
    plt.savefig(filename)

def SaveValues(filename, best_config, best_rmse, best_t_time, best_p_time, pred):
    with open(filename, 'w') as f1:
        f1.write('config: ' + str(best_config))
        f1.write('rmse: ' + str(best_rmse))
        f1.write('train_time: ' + str(best_t_time))
        f1.write('predict_time: ' + str(best_p_time))
        for val in pred:
            f1.write(str(val))

def GridSearch(data, n_test, data_name, configs, alg_name, alg_fuct, config_names):
    print("starting "+alg_name+" grid search")
    best_rmse = float("inf")
    best_p_time = None
    best_t_time = None
    best_p_values = None
    best_config = []
    with open('log/' + alg_name + '_' + data_name + '.csv', 'wb') as csvfile:
        metawriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        metawriter.writerow(['rmse', 'train_time', 'predict_time'] + config_names)
        print ("config, rmse, t_time, p_time")
        for config in configs:
            res = alg_fuct(data, n_test, config)
            if res != None:
                rmse, t_time, p_time, p_vals = res
                print (rmse, t_time, p_time, config)
                metawriter.writerow([rmse, t_time, p_time] + config)
                if rmse < best_rmse:
                    best_config, best_rmse, best_t_time, best_p_time, \
                    best_p_values = config, rmse, t_time, p_time, p_vals
            else:
                print(config, 'error')
        print('*********** best result is: ', best_config, best_rmse, 
            best_t_time, best_p_time, '**************')

    pred = best_p_values
    if best_p_time:
        n_in = best_config[0] if len(best_config)>0 else None
        if alg_name == 'sarima' or alg_name == 'ets':
            n_in = 0
        #print  "*****",  n_test, n_in, data, pred, "*****"
        PlotTest('images/'+data_name+'_'+alg_name+'_plottest.pdf', data, pred, n_test)
        PlotAll('images/'+data_name+'_'+alg_name+'_plotall.pdf', data, pred, n_test, n_in)
        SaveValues('images/'+data_name+'_'+alg_name+'_values.csv', best_config, best_rmse, best_t_time, best_p_time, pred)

    return best_config, best_rmse, best_t_time, best_p_time, pred

def RunExp(alg_name, alg_funct, alg_configs, config_names):
    print('shortterm')
    ts_len = 385
    n_test = 85

    print('datarates')
    colname = 'ave_rate_Mbps'
    data = LoadData("15min_dataset.csv", ts_len, colname)
    GridSearch(data, n_test, 'shortterm_datarate', alg_configs, alg_name, alg_funct, config_names)

    print('packet counts')
    colname = 'total_pkts'
    data = LoadData("15min_dataset.csv", ts_len, colname)
    GridSearch(data, n_test, 'shortterm_packetrate', alg_configs, alg_name, alg_funct, config_names)

    print('longterm')
    ts_len = 4343
    n_test = 843

    print('datarates')
    colname = 'ave_rate'
    data = LoadData("daily_datasetv2.csv", ts_len, colname)
    GridSearch(data, n_test, 'longterm_datarate', alg_configs, alg_name, alg_funct, config_names)

    print('packet counts')
    colname = 'total_pkts'
    data = LoadData("daily_datasetv2.csv", ts_len, colname)
    GridSearch(data, n_test, 'longterm_packetrate', alg_configs, alg_name, alg_funct, config_names)
