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
try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

def plot_confusion_matrix(cm, classes, classifier_name, save=False,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save:
        if normalize:
            save_name = base_dir + '/static/confusion_matrix/' + str(classifier_name) + '_norm.png'
        else:
            save_name = base_dir + '/static/confusion_matrix/' + str(classifier_name) + '.png'
        plt.savefig(save_name)
    plt.show()
    plt.close()

if __name__ == "__main__":
    '''
    Based on the https://github.com/theumairahmed/User-Identification-and-Classification-From-Walking-Activity/blob/master/learn.py
    '''
    feature_csv_dir_file = base_dir + '/data/processed/shallow_features.csv'
    dataset = np.loadtxt(feature_csv_dir_file, delimiter=",",skiprows=1)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    class_names = ['user-1', 'user-2', 'user-3',\
               'user-4', 'user-5', 'user-6',\
               'user-7','user-8','user-9',\
               'user-10','user-11','user-12'\
               ,'user-13','user-14','user-15'\
               ,'user-16','user-17','user-18']

    classifiers = {'RandomForestClassifier':RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),\
                'KNN':KNeighborsClassifier(10),\
                'LinearSVM':SVC(kernel="linear", C=0.025)}
    classifiers_title = list(classifiers.keys())
    scores=np.empty(10)
    means_scores=[]
    stddev_scores=[]

    X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.3) # 70-30 split

    for i in range(classifiers.__len__()):
        classifiers[classifiers_title[i]].fit (X_train,y_train)
        y_pred = classifiers[classifiers_title[i]].predict(X_test)
        scores = cross_val_score(classifiers[classifiers_title[i]],X,y,cv=10)
        cnf_matrix = confusion_matrix(y_test, y_pred)

        means_scores.append(scores.mean())
        stddev_scores.append(scores.std())

        print("[Results For ",classifiers_title[i], "] Mean: ",scores.mean()," Std Dev: ",scores.std())

        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix

        plt.figure(figsize=(8,8))
        classifier_name = classifiers_title[i]
        plot_confusion_matrix(cnf_matrix, classes=class_names, classifier_name=classifier_name, save=True,
                              title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plt.figure(figsize=(8,8))
        plot_confusion_matrix(cnf_matrix, classes=class_names, classifier_name=classifier_name, save=True,
                              normalize=True, title='Normalized confusion matrix')
        plt.show()
