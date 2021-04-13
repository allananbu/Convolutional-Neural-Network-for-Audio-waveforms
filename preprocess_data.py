# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:07:36 2021

@author: Allan
"""

import numpy as np
import librosa
import librosa.display
import os
import glob
import re

from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,Input,LSTM,TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend
from tensorflow.keras.layers import Activation

from tensorflow.keras.utils import to_categorical


L = 16000
def pad_audio(samples):
    if len(samples) >= L:
        return samples
    else:
        return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))
def chop_audio(samples,L=16000,num=20):
    for i in range(num):
        beg=np.random.randint(0,len(samples)-L)
        yield samples[beg:beg+L]

def log_specgram(audio,sample_rate,window_size=20,step_size=10,eps=1e-10):
    nperseg=int(round(window_size*sample_rate/1e3))
    noverlap=int(round(step_size*sample_rate/1e3))
    freqs,times,spec=signal.spectrogram(audio,fs=sample_rate,window='hann',nperseg=nperseg,noverlap=noverlap,detrend=False)
    return freqs,times,np.log(spec.T.astype(np.float32)+eps)



def one_hot_encode(class_name,classes):
    try:
        idx=classes.index(class_name)
        vec=np.zeros(len(classes))
        vec[idx]=1
        return vec
    except ValueError:
        return None
    


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

def preprocess_data(inpath='Data/',outpath='Processed/'):
#    if not os.path.exists(outpath):
#        os.mkdir(outpath,755)
    nb_classes = len(classes)
    y_data=[]
    x_data=[]
    for idx,cl in enumerate(classes):
        
        this_Y=np.array(one_hot_encode(cl,classes))
        this_Y=this_Y[np.newaxis,:]
#        if not os.path.exists(outpath+cl):
#            os.mkdir(outpath+cl,755)
        class_files = os.listdir(inpath+cl)
        for idx2,infilename in enumerate(class_files):
            audio_path = inpath+cl+'/'+infilename
            sample_rate, samples=wavfile.read(audio_path)
            samples=pad_audio(samples)
            if len(samples)>16000:
                n_samples=chop_audio(samples)
            else:
                n_samples=[samples]
            for samples in n_samples:
                resampled=signal.resample(samples,int(new_sample_rate/sample_rate*samples.shape[0]))
                _,_,specgram=log_specgram(resampled,sample_rate=new_sample_rate)
                y_data.append(idx)
                x_data.append(specgram)
    
    x_data=np.array(x_data)
    x_data=x_data.reshape(tuple(list(x_data.shape)+[1]))
    y_data=np.array(y_data)
    return x_data,y_data
#    y_train=

#            aud,sr=librosa.load(audio_path,sr=None)
#            melgram=librosa.amplitude_to_db(librosa.feature.melspectrogram(aud,sr=sr,n_mels=96))[np.newaxis,np.newaxis,:,:]
#            outfile=outpath+cl+'/'+infilename+'.npy'
#            np.save(outfile,melgram)
            

if __name__ == '__main__':
    classes = os.listdir('Data/')
    new_sample_rate = 8000
    global x_train
    global y_train
    x_data,y_data=preprocess_data()
    #x_data,y_data=shuffle_data(x_data,y_data)
#    x_data.resize([x_data.shape[1],x_data.shape[2],1,x_data.shape[0]])
#    y_data.resize([y_data.shape[1],y_data.shape[2],y_data.shape[0]])
    
    X_train, X_test, Y_train, Y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=42)
    np.save('X_train.npy',X_train)
    np.save('X_test.npy',X_test)
    np.save('Y_train.npy',Y_train)
    np.save('Y_test.npy',Y_test)
    