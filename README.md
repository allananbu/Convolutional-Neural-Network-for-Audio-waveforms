# Convolutional-Neural-Network-for-Audio-waveforms
TensorFlow trainin for Voice Commands [on/off] with Post-quantization training

Step 1 - Pre-process .wav files by resampling to 8KHz and obtain mel spectogram. Save Train & Test data along with their classes
Step 2 - Train CNN with three layers to obtain model. Test model with microphone recorded data.
Step 3 - Post-training quantization to obtain .tflite model
Step 4 - Verify model in MATLAB
