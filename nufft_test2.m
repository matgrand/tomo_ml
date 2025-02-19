%% Script for testing the nufft and nufftn functions
clc; clear all; close all;

%Signals
hx1 = 7; % first harmonic
hx2 = 12; % second harmonic 
phase = deg2rad(80);

% leave uncommented the one you want to test
% Signal = @(x) sin(2*pi*hx1*x + phase) + 0.8 * sin(2*pi*hx2*x + phase);  % WITH 2π
Signal = @(x) sin(hx1*x + phase) + 0.8 * sin(hx2*x + phase);  % WITHOUT 2π

disp("Signal: " + func2str(Signal) + " with hx1=" + hx1 + " and hx2=" + hx2);

N = 100; % number of samples of the signal

% length of the domain, 2π in the real case (toroidal length), but can be anything (try)
L = 2*pi; % full toroidal length
% L = 18; % random number to prove it can be anything

FS = N/L; % Sampling frequency N samples in L meters 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uniform signal
X = linspace(0, L, N); % x coordinates
S = Signal(X); % signal
Y = fft(S); % fft of the uniform grid signal
Y = 2*abs(fftshift(Y))/N; % normalize the fft

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% non-uniform signal
% Xnu = X; % x coordinates (same as X for reality check)
Xnu = [linspace(0, L/2, N/2), linspace(1.2*L/2, L, N/2)]; % non-uniformly sampled x coordinates (N samples)

Snu = Signal(Xnu); % non-uniformly sampled signal

%% nufft, uncomment the version u prefer
% nufft v1 -> change the x coordinates to be in [0,N] ( so that Fs=1hz, T=1s ) 
Ynu = nufft(Snu, FS*Xnu); % nufft

% % nufft v2 -> set the frequency axis 
% freqs = FS*(0:N-1)/N; % frequency axis for the nufft function (see https://www.mathworks.com/help/matlab/ref/double.nufft.html)
% Ynu = nufft(Snu, Xnu, freqs); % nufft

Ynu = 2*abs(fftshift(Ynu))/N; % normalize the nufft

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTTING 
figure('Name', 'NUFFT Test', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 600]);

f = 2*pi * FS * (-N/2:N/2-1)/N; % frequency axis (for plotting both fft and nufft) %NOTE THE 2π HERE

% uniform signal
subplot(2, 3, 1);
stem(X, S);
title('Uniform Signal');
% standard FFT
subplot(2, 3, 2);
stem(f, Y);
title('Standard FFT');
% standard FFT (second half)
subplot(2, 3, 3);
stem(f(N/2+1:N), Y(N/2+1:N));
title('Standard FFT (Positive Frequencies)');
% non-uniform signal
subplot(2, 3, 4);
stem(Xnu, Snu);
title('Non-uniformly Sampled Signal');
% NUFFT
subplot(2, 3, 5);
stem(f, Ynu);
title('NUFFT');
% NUFFT (second half)
subplot(2, 3, 6);
stem(f(N/2+1:N), Ynu(N/2+1:N));
title('NUFFT (Positive Frequencies)');
disp("done")
