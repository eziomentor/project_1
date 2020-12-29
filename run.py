#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask_restful import Resource
from flask import Flask, request
from flask import request, jsonify
from keras.models import Sequential
from flask_restful import Api, Resource, reqparse
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import pandas as pd
import joblib



model = keras.models.load_model('my_model.h5')
app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
     parser = reqparse.RequestParser()
     parser.add_argument('data', type=float)
     data = parser.parse_args()
     
    
     x_new = np.fromiter(data.values(), dtype=float)
     
        
        
     y_pred = model.predict(x_new)
     y_new = np.argmax(y_pred)
        
     if y_new == 0:
        condition = 'Healthy'
        return 'The Condition is {}'.format(condition)
     
     elif y_new == 1:
          condition = 'Inner race Faulty'
          return 'The Condition is {}'.format(condition)
     
     elif y_new == 2:
          condition = 'Outer race Faulty'
          return 'The Condition is {}'.format(condition)
            

if __name__ == '__main__':
     app.run(host='127.0.0.1', port=8000)

