# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:04:46 2021

@author: Allan
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras import backend
from keras.utils import np_utils
import os
from os.path import isfile

from timeit import default_timer as timer

classes = os.listdir('Data/')

def get_files(path='Processed/',train_percent=0.8):
    sum_train=0
    sum_test=0
    sum_total=0
    subdir=os.listdir(path)
    for subd in subdir:
        files=os.listdir(path+subd)
        no_files=len(files)
        sum_total+=no_files
        n_train=int(train_percent*no_files)
        n_test=no_files-n_train
        sum_train+=n_train
        sum_test+=n_test
    return sum_total,sum_train,sum_test

def get_sample_dimen(path='Processed/'):
    class_name=os.listdir(path)[0]
    files=os.listdir(path+class_name)
    in_file=files[0]
    audio_path=path+class_name+'/'+in_file
    melgram=np.load(audio_path)
    return melgram.shape

def one_hot_encode(class_name,classes):
    try:
        idx=classes.index(class_name)
        vec=np.zeros(len(classes))
        vec[idx]=1
        return vec
    except ValueError:
        return None
    

def build_datasets(train_percent=0.8,path='Processed/'):
    tot_files,tot_train,tot_test=get_files(path=path,train_percent=train_percent)
    
    no_class=len(classes)
    
    #pre-allocate memory for speed
    mels_dimen=get_sample_dimen(path=path) #get shape of every file
    X_train=np.zeros(tot_train,mels_dimen[1],mels_dimen[2],mels_dimen[3])
    X_test=np.zeros(tot_test,mels_dimen[1],mels_dimen[2],mels_dimen[3])
    Y_train=np.zeros(tot_train,no_class)
    Y_test=np.zeros(tot_test,no_class)
    
    path_train=[]
    path_test=[]
    
    train_count=0
    test_count=0
    
    for idx,class_name in enumerate(classes):
        this_Y=np.array(one_hot_encode(class_name,classes))
        this_Y=this_Y[np.newaxis,:]
        class_files=os.listdir(path+class_name)
        n_files=len(class_files)
        n_load=n_files
        n_train=int(train_percent*n_load)
        print_every=200
        print('')
        for idx2,file_name in enumerate(class_files[1:n_load]):
            audio_path=path+class_name+'/'+file_name
            if (0==idx2%print_every):
                print('\r Loading classes: {:14s}({:2d} of {:2d}classes)').format(class_name,idx+1,no_class),
                    ',file',idx2+1,'of',n_load,":",audio_path,sep='')
            
        
        