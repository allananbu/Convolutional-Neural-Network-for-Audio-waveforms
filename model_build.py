# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:25:11 2021

@author: Allan
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,Input,LSTM,TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend
from tensorflow.keras.layers import Activation

from tensorflow.keras.utils import to_categorical
import os
from os.path import isfile

X_train=np.load('X_train.npy')
X_test=np.load('X_test.npy')
Y_train=np.load('Y_train.npy')
Y_test=np.load('Y_test.npy')

X_train=X_train.reshape([96,32,1,3576])
X_test=X_test.reshape([96,32,1,894])

classes=['on','off']
no_class=len(classes)

no_filter=32
pool_size=(2,2)
kernel_size=(3,3)
no_layers=3
input_shape=(X_train.shape[0],X_train.shape[1],1)
    
model = Sequential()
model.add(Conv2D(8, 3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
  
model.add(Conv2D(16, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
  
model.add(Conv2D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
  
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.summary()
    
    #Initialize weights using checkpts
#load_checkpt=True
#checkpt_path='weights.hdf5'
#if load_checkpt:
#    print('Looking for previous weights')
#    if (isfile(checkpt_path)):
#        print('File detected, loading wgts')
#        model.load_weights(checkpt_path)
#    else:
#        print('No file detected, start from first')
#else:
#    print('start from scratch, no checkpts')
#    checkptr=ModelCheckpoint(filepath=checkpt_path,verbose=1,save_best_only=True)
    
    #train and score model
batch_size=128
no_epoch=30
model.fit(X_train,Y_train,batch_size=batch_size,epochs=no_epoch,verbose=1,
          validation_data=(X_test,Y_test))
score=model.evaluate(X_test,Y_test,verbose=0)
print('Test score',score[0])
print('Test accuracy',score[1])
