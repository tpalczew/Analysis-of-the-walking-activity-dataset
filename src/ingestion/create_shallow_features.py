import numpy as np
import csv
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ImportError:
    import pickle
import math
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools

try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")


def window(axis,dx=100):
    '''
    from https://github.com/theumairahmed/User-Identification-and-Classification-From-Walking-Activity/blob/master/Preprocessing.py
    '''
    start = 0;
    size = axis.count();
    while (start < size):
        end = start + dx
        yield start,end
        start = start+int (dx/2)

def window_summary(axis, start, end):
    '''
    from https://github.com/theumairahmed/User-Identification-and-Classification-From-Walking-Activity/blob/master/Preprocessing.py
    '''
    acf = stattools.acf(axis[start:end])
    acv = stattools.acovf(axis[start:end])
    sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
    return [
        axis[start:end].mean(),
        axis[start:end].std(),
        axis[start:end].var(),
        axis[start:end].min(),
        axis[start:end].max(),
        acf.mean(), # mean auto correlation
        acf.std(), # standard deviation auto correlation
        acv.mean(), # mean auto covariance
        acv.std(), # standard deviation auto covariance
        skew(axis[start:end]),
        kurtosis(axis[start:end]),
        math.sqrt(sqd_error.mean())
    ]

def features(user_id):
    '''
    from https://github.com/theumairahmed/User-Identification-and-Classification-From-Walking-Activity/blob/master/Preprocessing.py
    basic modification to get things right
    '''
    for (start, end) in window(user_id[0]): # 0 - Time
        features = []
        for axis in [1, 2, 3, 4]: # 1 - X, 2 - Y, 3 - Z, 4 - R
            features += window_summary(user_id[axis], start, end)
        yield features

if __name__ == "__main__":
    pickle_file_with_df = base_dir + '/data/preprocessed/dic_pd_df_t_x_y_z_r.pickle'
    df = pickle.load( open( pickle_file_with_df, "rb" ) )
    user_list = []
    for index, d in df.items():
        if index !=7 and index!=10 and index !=15 and index !=17:
            user_list.append([d['Time'], d['X'], d['Y'], d['Z'], d['R']])
            # if need here one can for example cut out a begining or end of the time-series by slicing

    feature_csv_dir_file = base_dir + '/data/processed/shallow_features.csv'
    with open(feature_csv_dir_file, 'w') as out:
        rows = csv.writer(out)
        for i in range(0, len(user_list)):
            for f in features(user_list[i]):
                rows.writerow([i]+f)
