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
import itertools
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from train_ml import plot_confusion_matrix
from create_shallow_features import window
try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def window_summary_deep(axis, start, end):
    '''
    for deep-learning; store whole time-series for each window
    '''
    vec = []
    for elem in axis[start:end]:
        vec.append(elem)
    return vec


def features_deep(user_id):
    '''
    from https://github.com/theumairahmed/User-Identification-and-Classification-From-Walking-Activity/blob/master/Preprocessing.py
    basic modification to get things right
    '''
    for (start, end) in window(user_id[0]): # 0 - Time
        features = []
        for axis in [1, 2, 3, 4]: # 1 - X, 2 - Y, 3 - Z, 4 - R
            features += window_summary_deep(user_id[axis], start, end)
        yield features


if __name__ == "__main__":
    '''
    Train Deep-Learning approach - LSTM
    '''

    # 1) Format data
    pickle_file_with_df = base_dir + '/data/preprocessed/dic_pd_df_t_x_y_z_r.pickle'
    df = pickle.load( open( pickle_file_with_df, "rb" ) )
    user_list = []
    for index, d in df.items():
        if index !=7 and index!=10 and index !=15 and index !=17:
            user_list.append([d['Time'], d['X'], d['Y'], d['Z'], d['R']])

    feature_csv_dir_file = base_dir + '/data/processed/deep_features.csv'
    with open(feature_csv_dir_file, 'w') as out:
        rows = csv.writer(out)
        for i in range(0, len(user_list)):
            for f in features_deep(user_list[i]):
                rows.writerow([i]+f)
