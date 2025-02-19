%% Script for testing the nufft and nufftn functions
clc; clear all; close all;

%Signals
hx1 = 3; % first harmonic
hx2 = 5; % second harmonic 
phase = deg2rad(80);

% leave uncommented the one you want to test
% Signal = @(x) 0.7*sin(2*pi*hx1*x + phase) + sin(2*pi*hx2*x + phase);  % WITH 2π
Signal = @(x) 0.7*sin(hx1*x + phase) + sin(hx2*x + phase);  % WITHOUT 2π

disp("Signal: " + func2str(Signal) + " with hx1=" + hx1 + " and hx2=" + hx2);

N = 100; % length of the signal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uniform signal
X = linspace(0, 2*pi, N); % x coordinates

% DX = X(2) - X(1); % delta x (Sampling period) (2π/N)
DX = 2*pi/N; % delta x (Sampling period) (2π/N)

S = Signal(X);
Y = fft(S); % fft of the uniform grid signal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% non-uniform signal
% Xnu = X; % x coordinates (same as X for reality check)
Xnu = [linspace(0, pi, N/2), linspace(1.2*pi, 2*pi, N/2)]; % non-uniformly sampled x coordinates (very important to have N samples)

Snu = Signal(Xnu); % non-uniformly sampled signal
% nufft, uncomment the version u prefer

%nufft v1 -> change the x coordinates to be in [0,N] ( so that Fs=1hz, T=1s ) 
Ynu = nufft(Snu, Xnu/DX); % nufft

% %nufft v2 -> set the frequency axis 
% freqs = (0:N-1)/N/DX; % frequency axis for the nufft function (see https://www.mathworks.com/help/matlab/ref/double.nufft.html)
% Ynu = nufft(Snu, Xnu, freqs); % nufft

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTTING 
figure;
n = (-N/2:N/2-1); % frequency axis (for plotting both fft and nufft)
% uniform signal
subplot(2, 2, 1);
stem(X, S);
title('Uniform Signal');
% standard FFT
subplot(2, 2, 2);
stem(n, 2*abs(fftshift(Y))/N);
title('Standard FFT');
% non-uniform signal
subplot(2, 2, 3);
stem(Xnu, Snu);
title('Non-uniformly Sampled Signal');
% NUFFT
subplot(2, 2, 4);
stem(n, 2*abs(fftshift(Ynu))/N);
title('NUFFT');
