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
try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")

from train_ml import plot_confusion_matrix
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


if __name__ == "__main__":
    '''
    Train Deep-Learning approach - LSTM
    '''

    # 1) load data
    feature_csv_dir_file = base_dir + '/data/processed/deep_features.csv'
    dataset = np.genfromtxt(feature_csv_dir_file, delimiter=",", filling_values=0, usecols=np.arange(0,401), invalid_raise = False)

    print(dataset)

    X = dataset[:, 1:]
    y = dataset[:, 0]
    class_names = ['user-1', 'user-2', 'user-3',\
               'user-4', 'user-5', 'user-6',\
               'user-7','user-8','user-9',\
               'user-10','user-11','user-12'\
               ,'user-13','user-14','user-15'\
               ,'user-16','user-17','user-18']

    X_train, X_temp, y_train, y_temp = train_test_split (X, y, test_size=0.2)
    X_test , X_valid, y_test, y_valid = train_test_split (X_temp,y_temp,test_size=0.5)

    print(X_train)

    X_train= np.reshape(X_train,(X_train.shape[0], 100, 4))
    # [samples, time steps, features] <-> [1715, 100, 4]

    X_test= np.reshape(X_test,(X_test.shape[0], 100, 4))

    X_valid= np.reshape(X_valid,(X_valid.shape[0], 100, 4))

    #y_train = np_utils.to_categorical(y_train)
    #y_test = np_utils.to_categorical(y_test)
    #y_valid = np_utils.to_categorical(y_valid)

    # 2) Architecture - dummy architecture just to present the basic approach based on keras

    model = Sequential()
    model.add(LSTM(32, input_shape=(100, 4), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(18, activation="sigmoid"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 3) Train

    early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)

    checkpoint = ModelCheckpoint('checkpooint.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1)

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                epochs=50,
                                batch_size=64,
                                callbacks=[early_stop, checkpoint])

    print(model.summary())

    # 4) Results

    y_pred = model.predict(X_valid)
    y_pred = np.array([np.argmax(y, axis=None, out=None) for y in y_pred])


    cnf_matrix = confusion_matrix(y_valid, y_pred)

    print(cnf_matrix)

    plt.figure(figsize=(8,8))
    classifier_name = 'LSTM'
    plot_confusion_matrix(cnf_matrix, classes=class_names, classifier_name=classifier_name, save=True,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(cnf_matrix, classes=class_names, classifier_name=classifier_name, save=True,
                          normalize=True, title='Normalized confusion matrix')
    plt.show()
