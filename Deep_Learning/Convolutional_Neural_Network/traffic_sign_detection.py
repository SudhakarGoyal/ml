#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:53:26 2018

@author: engineer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, exposure, transform
import glob
import os
from skimage import io
from keras.applications import InceptionV3

num_classes = 43
img_size = 32

def img_preprocessing(img):
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)
    
    return img
    
def get_class(img_path):
    return int(img_path.split('/')[-2])

root_dir  = '/home/engineer/Desktop/Traffic_Sign_Detection/GTSRB_training/Final_Training/Images/'
imgs = []
labels = []

imgs_path = glob.glob(os.path.join(root_dir,'*/*.ppm'))
np.random.shuffle(imgs_path)
for i in imgs_path:
# =============================================================================
#     img = img_preprocessing(io.imread(i))
# =============================================================================
    img = io.imread(i)
    label = get_class(i)
    img = img_preprocessing(img)
    img = color.rgb2gray(img)
    img = transform.resize(img, (img_size, img_size))
    imgs.append(img)    
    labels.append(label)
    
X = np.array(imgs)
X = X.reshape(X.shape[0],img_size,img_size,1)
Y = np.eye(num_classes)[labels]   #no need for one hot encoding

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint  

from sklearn.cross_validation import train_test_split
# =============================================================================
# inception_conv = InceptionV3(weights="imagenet")
# =============================================================================
X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                  test_size=0.2, random_state=42)


# =============================================================================
# 
# inception_conv.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.inception.hdf5', 
#                                verbose=1, save_best_only=True)
# 
# inception_conv.fit(X_train, Y_train, 
#           validation_data=(X_val, Y_val),
#           epochs=20, batch_size=32,  verbose=1)
# =============================================================================

from keras.layers import Input
input_img = Input(shape = (32, 32, 3))

def cnn_in():
    model1 = Conv2D(64,(1,1), padding='same', activation = 'relu')(input_img)
    model1 = Conv2D(64,(3,3), padding='same', activation='relu')(model1)
    
    model2 = (Conv2D(64,(1,1), padding='same', activation = 'relu'))(input_img)
    model2 = (Conv2D(64,(5,5), padding='same',activation = 'relu'))(model2)

    model3 = (MaxPooling2D((3,3), strides=(1,1), padding='same'))((input_img))
    model3 = (Conv2D(64,(1,1), padding='same', activation='relu'))(model3)

    output = concatenate([model1, model2, model3], axis = 3)
    
    model = (Flatten())(output)    
    model = Dense(num_classes,activation='softmax')(model)
    
    return model
    
#VGG 
def cnn_model():
   
    model = Sequential()
    model.add(Conv2D(64,(3,3), padding='same', input_shape=(32,32,1)))
# =============================================================================
#     model.add(Conv2D(64,(3,3), padding='same'))
# =============================================================================
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64,(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128,(3,3), padding='same'))
# =============================================================================
#     model.add(Conv2D(128,(3,3), padding='same'))
# =============================================================================
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64,(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(256,(3,3), padding='same'))
# =============================================================================
#     model.add(Conv2D(256,(3,3), padding='same'))
# =============================================================================
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64,(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))
    
# =============================================================================
#     model.add(Conv2D(512,(3,3), padding='same'))
#     model.add(Conv2D(512,(3,3), padding='same'))
#     model.add(Conv2D(512,(3,3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
# =============================================================================
# =============================================================================
#     model.add(Dropout(0.2))
# =============================================================================
# =============================================================================
#     
#     model.add(Conv2D(512,(3,3), padding='same'))
#     model.add(Conv2D(512,(3,3), padding='same'))
#     model.add(Conv2D(512,(3,3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
# =============================================================================
# =============================================================================
#     model.add(Dropout(0.2))
# =============================================================================
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))
        
    return model

# =============================================================================
# model = cnn_model()
    
# =============================================================================
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10.)

datagen.fit(X_train)

# =============================================================================
# model = Model(inputs = input_img, outputs = cnn_inception())
# =============================================================================
model = cnn_model()
model.summary()
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 20
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    epochs=epochs, validation_data=(X_val, Y_val),
                    callbacks=callbacks_list)
# =============================================================================
# history = model.fit(X,Y, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, validation_split=0.2,verbose=1)
# 
# =============================================================================
X_test =[]
Y_test = []
test_dataset = pd.read_csv("/home/engineer/Desktop/Traffic_Sign_Detection/GT-final_test.csv", sep=';')

for file_name, classId in zip(list(test_dataset['Filename']), list(test_dataset['ClassId'])):
    img_path = os.path.join("/home/engineer/Desktop/Traffic_Sign_Detection/GTSRB_test/Final_Test/Images/", file_name)
    img = io.imread(img_path)
    img = img_preprocessing(img)
    img = color.rgb2gray(img)
    img = transform.resize(img, (img_size, img_size))
    X_test.append(img)
    
    Y_test.append(classId)
    


X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0],img_size, img_size,1)
Y_test = np.array(Y_test)
model.load_weights("weights.best.hdf5")
Y_pred = model.predict_classes(X_test)
# =============================================================================
# Y_pred = model.predict(X_test)
# =============================================================================
sum_= 0 
for i in range(len(X_test)):
    if((Y_pred[i]) == Y_test[i]):
        sum_+=1

print(sum_/len(X_test))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#rgb image woth no histogram equalization with batch normalization gave an accuracy of 96.37%
#rgb with no histogram equalization with no batch normalization gave an accuracy of 94.72% and it involved early stopping 
#converting rgb into grayscale gave an accuracy of 95.87 on the test set
#equalizing histogram of Y and then converting the image in grayscale  gace an accuracy of 96.65 for 20 epochs. Accuracy would increase if epochs are increased as graph showed the same stuff
#equalizing histogram of Y and then converting the image in grayscale  gace an accuracy of 97.54 for 40 epochs. The model had 2 convnet layers together
