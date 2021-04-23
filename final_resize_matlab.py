# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:34:46 2021

@author: Allan
"""

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


x=scipy.io.loadmat('new_size_final.mat')
y=scipy.io.loadmat('new_size_label.mat')

x_data=x['P']
y_data=y['label']

y_data=to_categorical(y_data,2,dtype='float32')

x_data=np.expand_dims(x_data,axis=3)

def shuffle_data(X,Y):
    assert (X.shape[0]==Y.shape[0])
    idx=np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX=np.copy(X)
    newY=np.copy(Y)
    #newpath=paths
    for i in range(len(idx)):
        newX[i]=X[idx[i],:]
        newY[i]=Y[idx[i],:]
        #newpath[i]=paths[idx[i]]
    return newX,newY

if __name__ == '__main__':
    x_total,y_total=shuffle_data(x_data,y_data)
    X_train, X_test, Y_train, Y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=42)
    np.save('X_train.npy',X_train)
    np.save('X_test.npy',X_test)
    np.save('Y_train.npy',Y_train)
    np.save('Y_test.npy',Y_test)