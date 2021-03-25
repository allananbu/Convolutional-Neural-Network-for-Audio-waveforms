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

classes = os.listdir('Data/')

def preprocess_data(inpath='Data/',outpath='Processed/'):
    if not os.path.exists(outpath):
        os.mkdir(outpath,755)
    nb_classes = len(classes)
    for idx,cl in enumerate(classes):
        if not os.path.exists(outpath+cl):
            os.mkdir(outpath+cl,755)
        class_files = os.listdir(inpath+cl)
        for idx2,infilename in enumerate(class_files):
            audio_path = inpath+cl+'/'+infilename
            aud,sr=librosa.load(audio_path,sr=None)
            melgram=librosa.amplitude_to_db(librosa.feature.melspectrogram(aud,sr=sr,n_mels=96))[np.newaxis,np.newaxis,:,:]
            outfile=outpath+cl+'/'+infilename+'.npy'
            np.save(outfile,melgram)
            

if __name__ == '__main__':
    preprocess_data()
        
    
    