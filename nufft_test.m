%% Script for testing the nufft and nufftn functions
clc; clear all; close all;
FIGSIZE = [0 0 1000 600];

%% common parameters
% NOISE = 0.0; % noise level
NOISE = 1.0; % noise levels

PERCENT = 0.4; % percentage of samples to keep
PEAK_THRSH = 0.4; % threshold for peak detection

%% 1-D problem
disp('1-D problem');

%Signals
hx1 = 120; % first harmonic
hx2 = 50; % second harmonic 
% signalX = @(x) sin(2*pi*hx2*x); % true signal
signalX = @(x) 0.7*sin(2*pi*hx1*x) + sin(2*pi*hx2*x); % true signal
disp("Signal: " + func2str(signalX) + " with hx1=" + hx1 + " and hx2=" + hx2);

%full grid signal
FSx = 1000; % sampling frequency along x
Tx = 1/FSx; % sampling time along x
Lx = 2000; % length of the signal
X = [0:Lx-1]*Tx; % x coordinates
Fx = FSx/2*linspace(0,1,Lx/2+1); % frequency axis

S = signalX(X) + NOISE*randn(size(X)); % full grid signal

%plot true and noisy signals
figure('Position', FIGSIZE);
p = find(X >= 0.2 & X <= 0.3); % plot only a portion of the signal
stem(X(p), S(p), 's');
title('True signal'); xlabel('x'); ylabel('S(x)');

%standard fft
fftS = fft(S, Lx); % fft of the true signal
fftS = 2*abs(fftS(1:Lx/2+1))/Lx; % normalize the fft

%find peaks
disp("Peaks in the fft at: [" + num2str(Fx(find(fftS > PEAK_THRSH))) + "]");

%plot fft 
figure('Position', FIGSIZE); hold on;
stem(Fx, fftS, 's'); hold off;
title('FFT of true signal'); xlabel('f'); ylabel('|F(f)|');

%nufft
% Xnu = [0:Lx-1]*Tx; % x coordinates (same as X for reality check)
Xnu = randsample(X, round(PERCENT*Lx)); % non-uniformly sampled x coordinates (10%, try even smaller values)
Xnu = sort(Xnu) + Tx/2*randn(size(Xnu)); % sort and add some noise to the samples (less than half of the sampling time)
Snu = signalX(Xnu) + NOISE*randn(size(Xnu)); % true signal non-uniformly sampled

%plot
figure('Position', FIGSIZE);
p = find(Xnu >= 0.2 & Xnu <= 0.3); % plot only a portion of the signal (it's unsorted)
% p = 1:length(Xnu); % plot the whole signal
stem(Xnu(p), Snu(p), 's');
title(['True signal non-uniformly downsampled at ', num2str(100*size(Xnu, 2)/Lx), '%']); 
xlabel('x'); ylabel('S(x)');

%nufft
nufftS = nufft(Snu, Xnu, Fx); % nufft of the true signal
nufftS = 2*abs(nufftS(1:Lx/2+1))/Lx/PERCENT; % normalize the nufft

%find peaks
peaks = find(nufftS > PEAK_THRSH); % find peaks in the nufft
disp("Peaks in the nufft at: [" + num2str(Fx(find(nufftS > PEAK_THRSH))) + "]");

%plot nufft 
figure('Position', FIGSIZE); hold on;
% xline(hx1, ':k', 'LineWidth', 1.5); xline(hx2, ':k', 'LineWidth', 1.5);
stem(Fx, nufftS, 's'); hold off; 
title(['NUFFT with ', num2str(100*size(Xnu, 2)/Lx), '%']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-D problem
disp('2-D problem');

%Signals
hy1 = 69; % first harmonic
hy2 = 110; % second harmonic
% signalXY = @(x, y) sin(2*pi*hx2*x) + sin(2*pi*hy2*y); % true signal
% signalXY = @(x, y) sin(2*pi*(hx2*x + hy2*y)); % true signal
% signalXY = @(x, y) 0.7*sin(2*pi*hx1*x) + sin(2*pi*hx2*x) + 0.7*sin(2*pi*hy1*y) + sin(2*pi*hy2*y); % true signal
signalXY = @(x, y) 0.7*sin(2*pi*(hx1*x + hy1*y)) + sin(2*pi*(hx2*x + hy2*y)); % true signal

disp("Signal: " + func2str(signalXY) + " with hx1=" + hx1 + ", hx2=" + hx2 + ", hy1=" + hy1 + ", hy2=" + hy2);

%full grid signal
FSy = 1000; % sampling frequency along y
Ty = 1/FSy; % sampling time along y
Ly = 2000; % length of the signal
Y = [0:Ly-1]*Ty; % y coordinates
Fy = FSy/2*linspace(0,1,Ly/2+1); % frequency axis
[FXX, FYY] = meshgrid(Fx, Fy); % make a grid of x and y frequencies: sizes: FXX=(Lx/2+1, Ly/2+1), FYY=(Lx/2+1, Ly/2+1)

[XX, YY] = meshgrid(X, Y); % make a grid of x and y coordinates: sizes: XX=(Lx, Ly), YY=(Lx, Ly)
XY = [XX(:), YY(:)]; % make a grid of x and y coordinates: sizes: XY=(Lx*Ly, 2)

S = signalXY(XX, YY) + NOISE*randn(size(XX)); % true signal

%plot 
figure('Position', FIGSIZE);
p = find(XX(:) >= 0.2 & XX(:) <= 0.3 & YY(:) >= 0.2 & YY(:) <= 0.3); % plot only a portion of the signal
scatter(XX(p), YY(p), 50, S(p), 's', 'filled'); 
colorbar; title('True signal'); xlabel('x'); ylabel('y'); zlabel('S(x, y)'); axis equal;

%standard fft
fftS = fft2(S, Lx, Ly); % fft of the true signal
fftS = 2*abs(fftS(1:Lx/2+1, 1:Ly/2+1))/Lx/Ly; % normalize the fft

%find peaks
peaks = find(fftS > PEAK_THRSH); % find peaks in the fft
disp('Peaks in the fft2 at: '); for i = 1:size(peaks, 1) disp("(" + FXX(peaks(i)) + ", " + FYY(peaks(i)) + ")"); end

%plot fft
figure('Position', FIGSIZE); hold on;
imagesc(Fx, Fy, fftS); % look closely
scatter(FXX(peaks), FYY(peaks), 100, 'r', 'o'); % highlight peaks
colorbar; axis equal; hold off;
title('FFT of true signal (look closely)'); xlabel('f_x'); ylabel('f_y'); 

%nufftn
[iXnu, iYnu] = ind2sub([Lx Ly], randperm(Lx*Ly, round(PERCENT*Lx*Ly))'); % convert to indices of x and y
Xnu = X(iXnu)' + Tx/2*randn(size(iXnu)); % non-uniformly sampled x coordinates
Ynu = Y(iYnu)' + Ty/2*randn(size(iYnu)); % non-uniformly sampled y coordinates

Snu = signalXY(Xnu, Ynu) + NOISE*randn(size(Xnu)); % true signal non-uniformly sampled

%plot
figure('Position', FIGSIZE);
p = find(Xnu >= 0.2 & Xnu <= 0.3 & Ynu >= 0.2 & Ynu <= 0.3); % plot only a portion of the signal
scatter(Xnu(p), Ynu(p), 50, Snu(p), 's', 'filled'); % scatter plot with square markers for non-uniform sampling
title(['True signal non-uniformly downsampled at ', num2str(100*size(Xnu, 1)/(Lx*Ly)), '%']);
xlabel('x'); ylabel('y'); zlabel('S(x, y)'); axis equal; colorbar; 

% nufftn
nufft2S = nufftn(Snu, [Xnu, Ynu], {Fx, Fy}); % nufftn of the true signal
nufft2S = reshape(nufft2S, [Lx/2+1, Ly/2+1]); % reshape the nufft to the grid (NOTE: nufftn does not )
nufft2S = 2*abs(nufft2S)/(Lx*Ly)/PERCENT; % normalize the nufft

%find peaks
peaks = find(nufft2S > PEAK_THRSH); % find peaks in the nufft
disp('Peaks in the nufft2 at: '); for i = 1:size(peaks, 1) disp("(" + FXX(peaks(i)) + ", " + FYY(peaks(i)) + ")"); end

%plot nufft
figure('Position', FIGSIZE); hold on;
imagesc(Fx, Fy, nufft2S); % look closely
scatter(FXX(peaks), FYY(peaks), 100, 'r', 'o'); % highlight peaks
colorbar; axis equal; hold off;
title(['NUFFT with ', num2str(100*size(Xnu, 1)/(Lx*Ly)), '%']); xlabel('f_x'); ylabel('f_y');
