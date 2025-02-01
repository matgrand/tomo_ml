%% Script for testing the nufft and nufftn functions
clc; clear all; close all;
FIGSIZE = [0 0 1000 600];

%% common parameters
NOISE = 0.0; % noise level
% NOISE = 1.0; % noise levels

PERCENT = 0.1; % percentage of samples to keep
PEAK_THRSH = 0.4; % threshold for peak detection

%% 1-D problem
disp('1-D problem');

hx1 = 120; % first harmonic
hx2 = 50; % second harmonic 
% signalX = @(x) sin(2*pi*hx2*x); % true signal
signalX = @(x) 0.7*sin(2*pi*hx1*x) + sin(2*pi*hx2*x); % true signal

disp("Signal: " + func2str(signalX) + " with hx1=" + hx1 + " and hx2=" + hx2);

%true solution on regular grid
FSx = 1000; % sampling frequency along x
Tx = 1/FSx; % sampling time along x
Lx = 2000; % length of the signal
X = [0:Lx-1]*Tx; % x coordinates
Fx = FSx/2*linspace(0,1,Lx/2+1); % frequency axis

S = signalX(X) + NOISE*randn(size(X)); % true signal

%plot true and noisy signals
figure('Position', FIGSIZE);
p = find(X >= 0.2 & X <= 0.3); % plot only a portion of the signal
% p = 1:Lx; % plot the whole signal
stem(X(p), S(p), 's');
title('True signal'); xlabel('x'); ylabel('S(x)');

%standard fft
fftS = fft(S, Lx); % fft of the true signal
fftS = 2*abs(fftS(1:Lx/2+1))/Lx; % normalize the fft

peaks = find(fftS > PEAK_THRSH); % find peaks in the fft
disp("Peaks in the fft at: [" + num2str(Fx(find(fftS > PEAK_THRSH))) + "]");

%plot fft 
figure('Position', FIGSIZE); hold on;
xline(hx1, ':k', 'LineWidth', 1.5); xline(hx2, ':k', 'LineWidth', 1.5);
stem(Fx, fftS, 's'); hold off;
title('FFT of true signal'); xlabel('f'); ylabel('|F(f)|');

%nufft
% Xnu = [0:Lx-1]*Tx; % x coordinates (same as X for reality check)
Xnu = randsample(X, round(PERCENT*Lx)); % non-uniformly sampled x coordinates (10%, try even smaller values)
Snu = signalX(Xnu) + NOISE*randn(size(Xnu)); % true signal non-uniformly sampled

figure('Position', FIGSIZE);
p = find(Xnu >= 0.2 & Xnu <= 0.3); % plot only a portion of the signal (it's unsorted)
% p = 1:length(Xnu); % plot the whole signal
stem(Xnu(p), Snu(p), 's');
title(['True signal non-uniformly downsampled at ', num2str(100*size(Xnu, 2)/Lx), '%']); 
xlabel('x'); ylabel('S(x)');

nufftS = nufft(Snu, Xnu, Fx); % nufft of the true signal
nufftS = 2*abs(nufftS(1:Lx/2+1))/Lx/PERCENT; % normalize the nufft

peaks = find(nufftS > PEAK_THRSH); % find peaks in the nufft
disp("Peaks in the nufft at: [" + num2str(Fx(find(nufftS > PEAK_THRSH))) + "]");

%plot nufft 
figure('Position', FIGSIZE); hold on;
xline(hx1, ':k', 'LineWidth', 1.5); xline(hx2, ':k', 'LineWidth', 1.5);
stem(Fx, nufftS, 's'); hold off;
title(['NUFFT with ', num2str(100*size(Xnu, 2)/Lx), '%']);


%% 2-D problem
disp('2-D problem');

hy1 = 69; % first harmonic
hy2 = 110; % second harmonic

signalXY = @(x, y) sin(2*pi*hx2*x) + sin(2*pi*hy2*y); % true signal
% signalXY = @(x, y) sin(2*pi*(hx2*x + hy2*y)); % true signal
% signalXY = @(x, y) 0.7*sin(2*pi*hx1*x) + sin(2*pi*hx2*x) + 0.7*sin(2*pi*hy1*y) + sin(2*pi*hy2*y); % true signal
% signalXY = @(x, y) 0.7*sin(2*pi*(hx1*x + hy1*y)) + sin(2*pi*(hx2*x + hy2*y)); % true signal

disp("Signal: " + func2str(signalXY) + " with hx1=" + hx1 + ", hx2=" + hx2 + ", hy1=" + hy1 + ", hy2=" + hy2);

%true solution on regular grid
FSy = 1000; % sampling frequency along y
Ty = 1/FSy; % sampling time along y
Ly = 2000; % length of the signal
Y = [0:Ly-1]*Ty; % y coordinates
Fy = FSy/2*linspace(0,1,Ly/2+1); % frequency axis

[XX, YY] = meshgrid(X, Y); % make a grid of x and y coordinates: sizes: XX=(Lx, Ly), YY=(Lx, Ly)
[FXX, FYY] = meshgrid(Fx, Fy); % make a grid of x and y frequencies: sizes: FXX=(Lx/2+1, Ly/2+1), FYY=(Lx/2+1, Ly/2+1)

S = signalXY(XX, YY) + NOISE*randn(size(XX)); % true signal

%plot true and noisy signals
figure('Position', FIGSIZE);
px = find(X >= 0.2 & X <= 0.3); py = find(Y >= 0.2 & Y <= 0.3); % plot only a portion of the signal
% px = 1:Lx; py = 1:Ly; % plot the whole signal
imagesc(X(px), Y(py), S(px, py));
% contourf(X(px), Y(py), S(px, py), 20); % contour plot is heavy with full signal
colorbar; title('True signal'); xlabel('x'); ylabel('y'); zlabel('S(x, y)');

%standard fft
fftS = fft2(S, Lx, Ly); % fft of the true signal
fftS = 2*abs(fftS(1:Lx/2+1, 1:Ly/2+1))/Lx/Ly; % normalize the fft

%find peaks
peaks = find(fftS > PEAK_THRSH); % find peaks in the fft
disp('Peaks in the fft2 at: ');
for i = 1:size(peaks, 1)
    disp("(" + FXX(peaks(i)) + ", " + FYY(peaks(i)) + ")");
end
% %plot fft
% figure('Position', FIGSIZE); hold on;
% imagesc(Fx, Fy, fftS); % look closely at the edges
% colorbar; hold off
% title('FFT of true signal (look closely)'); xlabel('f_x'); ylabel('f_y'); 

