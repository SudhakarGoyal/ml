#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:08:24 2018

@author: engineer
"""

import numpy as np
import keras 
from keras.datasets import imdb
from collections import Counter
from keras import Sequential
from keras.preprocessing.text import Tokenizer

(x_train, y_train), (x_test, y_test) = imdb.load_data()
tokenizer = Tokenizer(num_words=500)

x_train = tokenizer.sequences_to_matrix(x_train, mode='binary' ) #we just need 0 and 1 
x_test = tokenizer.sequences_to_matrix(x_test,mode='binary')

y_train = keras.utils.to_categorical(y_train,2)
y_test = keras.utils.to_categorical(y_test,2)


from keras.layers.core import Dense, Dropout, Activation
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(500,)))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,epochs = 20, verbose=1, batch_size=64)
score = model.evaluate(x_train,y_train)
print("acc train", score[1])
score = model.evaluate(x_test,y_test)
print("acc test ", score[1])