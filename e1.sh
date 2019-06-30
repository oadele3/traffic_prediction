#!/bin/bash
python esn.py | tee log/esn_stdout.txt
python cnn.py | tee log/cnn_stdout.txt
python lstm.py | tee log/lstm_stdout.txt
python sarima.py | tee log/sarima_stdout.txt
python ets.py | tee log/ets_stdout.txt
