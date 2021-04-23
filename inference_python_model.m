clc;
close all;
clear all;
%----------Load the Model--------------------------------------------

net=importKerasNetwork('audio_recog_final.h5');


%---------Test Input fron Dataset------------------------------------

%--DataSetLink: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

%[x, fs]=audioread('E:\Research\CNN_Quantization_Paper1\VoiceActivityDetection - Copy\speech_commands_v0.01\six\0cd323ec_nohash_0.wav');
%--Dataset Sampling Rate is 16000Hz----------------------------------
%x = downsample(x,2);

%------------Individual RealTime acquired Test Input-----------------
load sig1.mat;

x=sig(1:8000)';
fs=8000;
t=(0:1:7999)*(1/fs);
figure(1)
subplot(1,2,1)
 plot(t,x);
 xlabel("Time(s)");
 ylabel("Amplitude(mV)");
 axis tight

fs=8000;
N = 240;
hopLength = 200;
fftLength=256;
% Buffer into frames.
y = audio.internal.buffer(x,N,hopLength);
Y = fft(y,fftLength,1);
Y = abs(Y);
xx=Y([2:129],:);
%Y = Y.^2;
for ii=0:31
P(ii+1,:)=sum(xx(((ii*4)+1):((ii+1)*4),:))/4;
end
spec=log10(P);
 specMin=min(min(spec));
 specMax=max(max(spec));
 subplot(1,2,2)
 pcolor(spec)
 %caxis([specMin+2 specMax])
 xlabel("Frame Number");
 ylabel("Log-Frequency Bin");
 shading flat
% [YPredicted,probs] = classify(net,spec,'ExecutionEnvironment','cpu');
% YPredicted
% probs