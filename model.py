#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
from flask import request, jsonify
from keras.models import Sequential
from flask_restful import Api, Resource, reqparse
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import joblib
import pickle
import json
from flask_restful import Resource


def model_f(X,Y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = Sequential()
    model.add(Dense(10, input_shape=(1,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(Adam(lr=0.001), 'categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=1)
    
    return model

if  __name__ == '__main__':
    
    inner = pd.read_csv('Faulty_inner.csv')
    outer = pd.read_csv('Faulty_outer.csv')
    healthy = pd.read_csv('healthy.csv')
    
    pd.DataFrame(healthy)
    pd.DataFrame(outer)
    pd.DataFrame(inner)
    
    merge = pd.merge(inner, outer, how="outer")
    dataset = pd.merge(merge, healthy, how="outer")
    
    X = dataset.Vibration
    x_norm = (X - np.min(X))/(np.max(X)-np.min(X))
    X = x_norm
    
    

    Y = pd.get_dummies(dataset.Condition)
    Y = Y.values
    
    mdl = model_f(X,Y)
    
mdl.save('my_model.h5')

