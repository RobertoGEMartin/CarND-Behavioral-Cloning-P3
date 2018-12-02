#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:08:35 2017

@author: goodfellow
"""
import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)
        
images = []
steering = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path= './data/IMG' + filename
    image = cv2.imread(current_path)
    images.append(image)
    steer_value = float(line[3])
    steering.append(steer_value)
    

X_train = np.array(images)
Y_train = np.array(steering)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras import losses

model = Sequential()
model.add(Flatten(input_shape=(320,160,3)))
model.add(Dense(1, activation=None, use_bias=True, 
                kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                kernel_regularizer=None, bias_regularizer=None))
model.compile(loss='mse', optimizer='adam', metrics=None, sample_weight_mode=None)
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1, callbacks=None, 
          validation_split=0.2, validation_data=None, 
          shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,)
model.save('model-p3-v00.h5')