# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:47:28 2023

@author: adina_l1uzsjt
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import random
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from faker import Faker as fake
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision, Recall, AUC
from imblearn.over_sampling import SMOTE


def read_data():
    #read in dataset
    path = 'cardio_train.csv'
    df = pd.read_csv(path,delimiter=";")
    x1=df.loc[:,df.columns!='cardio']
    y1=df.loc[:,'cardio']
    return x1,y1


def train_rf(x1,y1, training_size=0.10):
    #do the train test split
    xtrain,xtest,ytrain,ytest=train_test_split(x1,y1, train_size=training_size, random_state=5)
    #oversample the data using SMOTE
    oversampler = SMOTE(random_state=30)
    #create the resampled X and Y training data
    X_train_resampled, Y_train_resampled = oversampler.fit_resample(xtrain,ytrain)
    #run Random Forest Classifier
    model = ensemble.RandomForestClassifier()
    model.fit(X_train_resampled, np.ravel(Y_train_resampled))
    y_pred_rfc = model.predict(xtest)
    accuracy = round(accuracy_score(ytest,y_pred_rfc)*100,3)
    return accuracy, ytest, y_pred_rfc


def main():
    X_train_resampled, Y_train_resampled = read_data()
    columns = ["Training Size", "Accuracy"]
    scores_df = pd.DataFrame(columns=columns)
    #fig, axes = plt.subplots(5, 4, figsize=(12, 15))

    for ts in range(1, 20, 1):
        ts_rounded = round(ts * 0.05,2)
        print(ts)
        acc_rf, ytest_rf, predict_rf = train_rf(X_train_resampled, Y_train_resampled, ts_rounded)
        scores_df = pd.concat([scores_df, pd.DataFrame({"Training Size": [ts_rounded], "Accuracy": [acc_rf]})], ignore_index=True)
        print(scores_df)

    return

if __name__ == '__main__':
    main()