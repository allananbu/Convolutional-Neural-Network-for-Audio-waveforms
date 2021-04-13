# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:10:05 2021

@author: Allan
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,Input,LSTM,TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt


#load dataset

X_train=np.load('X_train.npy')
X_test=np.load('X_test.npy')
Y_train=np.load('Y_train.npy')
Y_test=np.load('Y_test.npy')

#define the classes
train_data=tf.cast(X_train,tf.float32)
val_data=tf.cast(X_test,tf.float32)

##store no of images in train & validation 


shape_1=99
shape_2=81

#Create CNN model

model = Sequential()
model.add(Conv2D(16,(3,3),padding='same', input_shape=(shape_1,shape_2,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(32,(3,3),padding='same', input_shape=(shape_1,shape_2,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(64,(3,3),padding='same', input_shape=(shape_1,shape_2,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation('softmax'))


#compile model
#Compile the model 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#fit for train data
hist=model.fit(train_data,Y_train,batch_size=250,epochs=15,validation_data=(val_data,Y_test))
score=model.evaluate(val_data,Y_test)
print('Test accuracy',score[1])
model.summary()

#plots
fig=plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper left')
plt.show()


model.save('audio_recog.h5')
