# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:33:20 2021

@author: Allan
"""

import numpy as np
import librosa
import librosa.display
import os
import glob
#from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

path='Processed/'

classes=['on','off']
no_class=len(classes)

sum_train=0
sum_test=0
sum_total=0
train_percent=0.8
subdir=os.listdir(path)
for cl in subdir:
    inpath=path+cl
    files=os.listdir(inpath)
    no_files=len(files)
    sum_total+=no_files
    n_train=int(train_percent*no_files)
    n_test=no_files-n_train
    sum_train+=n_train
    sum_test+=n_test

tot_files=sum_total
tot_train=sum_train
tot_test=sum_test
#    
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
#
mels_dimen=get_sample_dimen()
X_tot=np.zeros([tot_files,mels_dimen[1],mels_dimen[2],mels_dimen[3]])
Y_tot=np.zeros([tot_files,no_class])
Y_tot.astype(str)
X_train=np.zeros([tot_train,mels_dimen[1],mels_dimen[2],mels_dimen[3]])
X_test=np.zeros([tot_test,mels_dimen[1],mels_dimen[2],mels_dimen[3]])
#Y_train=np.zeros([tot_train,:])
#Y_test=np.zeros([tot_test,:])
#Y_train=[]
#Y_test=[]
train_count=0
test_count=0
tot_count=0

idx2=0
path_train=[]
path_test=[]
#
for idx,class_name in enumerate(classes):
    this_Y=np.array(one_hot_encode(class_name,classes))
    this_Y=this_Y[np.newaxis,:]
    class_files=os.listdir(path+class_name)
    n_files=len(class_files)
    n_load=n_files
    n_train=int(train_percent*n_load)
    print_every=200
    #print('')
    for ide,file_name in enumerate(class_files[1:n_load]):
        audio_path=path+class_name+'/'+file_name
#        if (0==idx2%print_every):
            
#            print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(class_name,idx+1,no_class),
#                       ", file ",idx2+1," of ",n_load,": ",audio_path,sep="")
            
#            aud,sr=librosa.load(audio_path,mono=mono,sr=None)
#            melgram=librosa.amplitude_to_db(librosa.feature.melspectrogram(aud,sr=sr,n_mels=96))[np.newaxis,np.newaxis,:,:]
#            melgram=melgram[:,:,:0:mels_dimen[3]] #clip upto 1st file size
        melgram=np.load(audio_path)
        idx2+=1
        
        X_tot[tot_count,:,:]=melgram
        Y_tot[tot_count,:]=this_Y
        tot_count+=1
#        if idx2<n_train:
#            X_train[train_count,:,:]=melgram
#            Y_train[train_count,:]=class_name
#            #Y_train.append(class_name)
#            #path_train.append(audio_path)
#            train_count+=1
#        else:
#            X_test[test_count,:,:]=melgram
#            Y_test[test_count,:]=class_name
##            Y_test.append(class_name)
#            #path_test.append(audio_path)
#            test_count+=1
#            print('')
            
#            print('Shuffle data order')
            
#        X_train,Y_train=shuffle_data(X_train,Y_train,path_train)
#        X_test,Y_test=shuffle_data(X_test,Y_test,path_test)
#
#
##Build model
#        
X_tot=np.delete(X_tot,(4471),axis=0)
X_tot=np.delete(X_tot,(4470),axis=0)
Y_tot=np.delete(Y_tot,(4470),axis=0)
Y_tot=np.delete(Y_tot,(4471),axis=0)

X_tot,Y_tot=shuffle_data(X_tot,Y_tot)

X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=0.2,random_state=42)

np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)
np.save('Y_train.npy',Y_train)
np.save('Y_test.npy',Y_test)

