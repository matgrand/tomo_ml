%% Script for testing the nufft and nufftn functions
clc; clear all; close all;
FIGSIZE = [0 0 1000 600];

%% common parameters
% NOISE = 0.0; % noise level
NOISE = 1.0; % noise levels

PERCENT = 0.2; % percentage of samples to keep
PEAK_THRSH = 0.4; % threshold for peak detection

%% 1-D problem
disp('1-D problem');

%Signals
hx1 = 120; % first harmonic
hx2 = 50; % second harmonic 
% signalX = @(x) sin(2*pi*hx2*x); % 1 harmonic
signalX = @(x) 0.7*sin(2*pi*hx1*x) + sin(2*pi*hx2*x);  % 2 harmonics
disp("Signal: " + func2str(signalX) + " with hx1=" + hx1 + " and hx2=" + hx2);

%Setting up the problem
FSx = 1000; % sampling frequency along x
Tx = 1/FSx; % sampling time along x
Lx = 2000; % length of the signal
Fx = FSx/2*linspace(0,1,Lx/2+1); % frequency axis

%nufft
% Xnu = [0:Lx-1]*Tx; % x coordinates (uniform for reality check)
Xnu = (Lx-1)*rand(1, round(PERCENT*Lx))*Tx; % non-uniformly sampled x coordinates
Snu = signalX(Xnu) + NOISE*randn(size(Xnu)); % non-uniformly sampled signal

%plot
figure('Position', FIGSIZE); 
p = find(Xnu >= 0.2 & Xnu <= 0.3); % plot only a portion of the signal
stem(Xnu(p), Snu(p), 's');
title(['Signal non-uniformly downsampled at ', num2str(100*size(Xnu, 2)/Lx), '%']); xlabel('x'); ylabel('S(x)');

%nufft
nufftS = nufft(Snu, Xnu, Fx); % nufft
nufftS = 2*abs(nufftS(1:Lx/2+1))/Lx/PERCENT; % normalize the nufft

%find peaks
disp("Peaks in the nufft at: [" + num2str(Fx(find(nufftS > PEAK_THRSH))) + "]");

%plot nufft 
figure('Position', FIGSIZE); stem(Fx, nufftS, 's'); title(['NUFFT with ', num2str(100*size(Xnu, 2)/Lx), '%']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-D problem
disp('2-D problem');

%Signals
hy1 = 69; % first harmonic
hy2 = 110; % second harmonic
% signalXY = @(x, y) sin(2*pi*hx2*x) + sin(2*pi*hy2*y); % 1 harmonic x + 1 harmonic y
% signalXY = @(x, y) sin(2*pi*(hx2*x + hy2*y)); % 1 combined harmonic
% signalXY = @(x, y) 0.7*sin(2*pi*hx1*x) + sin(2*pi*hx2*x) + 0.7*sin(2*pi*hy1*y) + sin(2*pi*hy2*y); % 2 harmonics x + 2 harmonics y
% signalXY = @(x, y) 0.7*sin(2*pi*(hx1*x + hy1*y)) + sin(2*pi*(hx2*x + hy2*y)); % 2 combined harmonics
signalXY = @(x, y) 0.7*sin(2*pi*(hx1*x + hy1*y)) + sin(2*pi*(hx2*x + hy2*y)) + 0.8*sin(2*pi*hy1*y); % 2 combined harmonics + 1 harmonic y
disp("Signal: " + func2str(signalXY) + " with hx1=" + hx1 + ", hx2=" + hx2 + ", hy1=" + hy1 + ", hy2=" + hy2);

%setting up the problem
FSy = 1000; % sampling frequency along y
Ty = 1/FSy; % sampling time along y
Ly = 2000; % length of the signal
Fy = FSy/2*linspace(0,1,Ly/2+1); % frequency axis
[FXX, FYY] = meshgrid(Fx, Fy); % make a grid of x and y frequencies: sizes: FXX=(Lx/2+1, Ly/2+1), FYY=(Lx/2+1, Ly/2+1)

Xnu = ((Lx-1)*rand(1, round(PERCENT*Lx*Ly))*Tx)'; % non-uniformly sampled x coordinates
Ynu = ((Ly-1)*rand(1, round(PERCENT*Lx*Ly))*Ty)'; % non-uniformly sampled y coordinates

% non-uniformly sampled 2-D signal
Snu = signalXY(Xnu, Ynu) + NOISE*randn(size(Xnu)); 

%plot
figure('Position', FIGSIZE); 
p = find(Xnu >= 0.2 & Xnu <= 0.3 & Ynu >= 0.2 & Ynu <= 0.3); % plot only a portion of the signal
scatter(Xnu(p), Ynu(p), 50, Snu(p), 's', 'filled'); % scatter plot with square markers for non-uniform sampling
title(['Non-uniformly sampled signal ', num2str(100*size(Xnu, 1)/(Lx*Ly)), '%']); xlabel('x'); ylabel('y'); zlabel('S(x, y)'); axis equal; colorbar; 

% nufftn
nufft2S = nufftn(Snu, [Xnu, Ynu], {Fx, Fy}); % nufftn
nufft2S = reshape(nufft2S, [Lx/2+1, Ly/2+1])'; % reshape the nufft to the grid (NOTE: nufftn already keeps only the first half of the frequencies)
nufft2S = 2*abs(nufft2S)/(Lx*Ly)/PERCENT; % normalize the nufft

%find peaks
peaks = find(nufft2S > PEAK_THRSH); % find peaks in the nufft
disp('Peaks in the nufft2 at: '); for i = 1:size(peaks, 1) disp("(" + FXX(peaks(i)) + ", " + FYY(peaks(i)) + ")"); end

%plot nufft
figure('Position', FIGSIZE); hold on;
imagesc(Fx, Fy, nufft2S); % look closely
scatter(FXX(peaks), FYY(peaks), 100, 'r', 'o'); % highlight peaks
title(['NUFFTN with ', num2str(100*size(Xnu, 1)/(Lx*Ly)), '%']); xlabel('f_x'); ylabel('f_y'); colorbar; axis equal; 
