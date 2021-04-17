# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 13:20:09 2021

@author: Allan
"""

import tensorflow as tf
import pathlib
import numpy as np

X_train=np.load('X_train.npy')
X_test=np.load('X_test.npy')
Y_train=np.load('Y_train.npy')
Y_test=np.load('Y_test.npy')

Ytrain=Y_train[:,1]
Ytest=Y_test[:,1]

#test_1=np.load('test_caps.npy',allow_pickle='TRUE').item()
#X_test=test_1['test']
#Ytest=test_1['Ytest']
#Ytest=np.array(Ytest)

#train_data=train_data/255
audio=(X_train,Ytrain)
img=tf.cast(audio[0],tf.float32)
font_ds=tf.data.Dataset.from_tensor_slices((img)).batch(1)


def representative_data_gen():
  for input_value in font_ds.take(500):
    # Model has only one input so each data point has one element.
    yield [input_value]

model=tf.keras.models.load_model('Audio_recog.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

#representative dataset
converter.representative_dataset = representative_data_gen

#convert with quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

#tf_model_dir=pathlib.Path('/Model/')
#tf_model_dir.mkdir(exist_ok=True,parents=True)
#tf_model_file=tf_model_dir/"Audio_recog_quant.tflite"
#tf_model_file.write_bytes(tflite_model)

tflite_model_files = pathlib.Path('Audio_recog_quant.tflite')
tflite_model_files.write_bytes(tflite_model)

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_files))


input_index_quant = interpreter.get_input_details()[0]["index"]
output_index_quant = interpreter.get_output_details()[0]["index"]
interpreter.allocate_tensors()

p=X_test.shape[0]
result_q=[]
#test quantized model
for i in range(0,p):
    image1=(X_test[i,:,:,:])
    
    #image1=image1.reshape(1,20,20,1)
    image1=np.expand_dims(image1,axis=0)
    test_im = tf.cast(image1,tf.float32)
    #test_image=test_image[:,:,:,0]

    interpreter.set_tensor(input_index_quant, test_im)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_index_quant)
    result_quant=np.argmax(predictions)
    result_q.append(result_quant)
result_q=np.array(result_q)
result_q=result_q.reshape(p,1)
Ytest=Ytest.reshape(p,1)
final_q=np.subtract(Ytest,result_q)
final_perq=np.where(np.nonzero(final_q))