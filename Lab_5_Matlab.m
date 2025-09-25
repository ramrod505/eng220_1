% TONE LAB: explore frequency, duration, and sampling rate
% - How to synthesize a sine wave
% - Nyquist/sampling rate intuition
% - Spectrogram vs waveform
%changes made

clc; clear; close all;

% --- Inputs (with simple defaults) ---
f = input('Frequency in Hz (e.g., 440): ');
if isempty(f) || ~isnumeric(f) || isnan(f), f = 440; end

dur = input('Duration in seconds (e.g., 1): ');
if isempty(dur) || ~isnumeric(dur) || isnan(dur), dur = 1; end

fs = input('Sampling rate in Hz (e.g., 44100): ');
if isempty(fs) || ~isnumeric(fs) || isnan(fs), fs = 44100; end

% --- Generate tone ---
t = 0:1/fs:dur;
y = sin(2*pi*f*t);

% Normalize for safety
y = y / max(abs(y)+eps); % eps handles the zero division error

% --- Play sound ---
disp('Playing tone...'); 
sound(y, fs);

% --- Plot waveform and spectrogram ---
figure('Name','Tone Lab','NumberTitle','off');
subplot(2,1,1);
plot(t, y), grid on
xlabel('Time (s)'), ylabel('Amplitude')
title(sprintf('Sine wave: f = %g Hz, fs = %g Hz', f, fs));

subplot(2,1,2);
% Use short window for better time resolution
win = round(0.03*fs); if win < 32, win = 32; end
noverlap = round(win*0.75);
nfft = max(256, 2^nextpow2(win));
spectrogram(y, win, noverlap, nfft, fs, 'yaxis');
title('Spectrogram');
colormap turbo; colorbar;

%% % REACTION TIME GAME: press Enter when you hear the beep
% - tic/toc timing, randomness, loops
% - Simple sound synthesis and basic plotting

clc; clear; close all;

fprintf(['Instructions:\n' ...
    '1) You will hear %d beeps at random times.\n' ...
    '2) As soon as you hear a beep, press Enter.\n\n'], 5);

N = 5;                              % number of trials
fs = 44100;                         % sample rate for the beep
beep_freq = 880;                    % Hz
beep_dur = 0.12;                    % seconds
delay_range = [0.7, 2.5];           % random delay before beep

rt = zeros(1,N);                    % reaction times (s)
for k = 1:N
    fprintf('Trial %d/%d: get ready...\n', k, N);
    pause(rand()*(delay_range(2)-delay_range(1)) + delay_range(1));

    % synthesize a short beep
    t = 0:1/fs:beep_dur;
    y = sin(2*pi*beep_freq*t);
    y = y / max(abs(y)+eps); % eps handles the zero division error
    sound(y, fs);

    tic;
    input('Press Enter NOW! ','s');
    rt(k) = toc;
    fprintf('Your reaction time: %.3f s\n\n', rt(k));
end

% Summary stats and plot
mean_rt = mean(rt);
fprintf('Average reaction time: %.3f s (std = %.3f s)\n', mean_rt, std(rt));

figure('Name','Reaction Times','NumberTitle','off');
bar(1:N, rt);
grid on
xlabel('Trial'), ylabel('Reaction time (s)')
title(sprintf('Reaction Times (mean = %.3f s)', mean_rt));

%% % CLICK POLYGON AREA: draw a shape and compute its area
% - ginput for simple interactivity
% - Vectors, polyarea, plotting

clc; clear; close all;

nmax = 100; % safety cap
figure('Name','Click Polygon Area','NumberTitle','off');
axis([0 10 0 10]); axis square; grid on
title({'Click points to outline a polygon','Press Enter when done'})
xlabel('x'), ylabel('y');

% Collect clicks
[x, y] = ginput(nmax);

if numel(x) < 3
    disp('Need at least 3 points to form a polygon.');
    return;
end

% Close the polygon
x_closed = [x; x(1)];
y_closed = [y; y(1)];

A = polyarea(x, y);  % signed area magnitude

% Plot
plot(x_closed, y_closed, '-o','LineWidth',1.5); hold on
fill(x, y, 'k', 'FaceAlpha', 0.05, 'EdgeColor','none');
scatter(x, y, 36, 'filled'); hold off
grid on
title(sprintf('Polygon area = %.3f square units', A));

%% % IMAGE EDGES: grayscale + simple Sobel edge detection (no toolboxes required)
% - imread, rgb2gray (or manual)
% - Convolution filters, gradient magnitude
% - Simple binary edge map

clc; clear; close all;

% Try to load a sample image; fall back if unavailable
try
    I = imread('peppers.png');
catch
    try
        I = imread('cameraman.tif');
    catch
        % generate synthetic image if samples are missing
        [X,Y] = meshgrid(linspace(-3,3,256));
        I = uint8( 255*(exp(-(X.^2+Y.^2)) + 0.2*rand(size(X))) );
    end
end

% Convert to grayscale if needed
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

% --- Edge detection using Sobel filters ---
fx = [-1 0 1; -2 0 2; -1 0 1];   % Sobel filter (x)
fy = fx';                        % Sobel filter (y)

Ix = conv2(double(Igray), fx, 'same');  % x-gradient
Iy = conv2(double(Igray), fy, 'same');  % y-gradient

Gmag = sqrt(Ix.^2 + Iy.^2);            % gradient magnitude

% Normalize and threshold
thresh = 0.2 * max(Gmag(:));           % threshold relative to max
E = Gmag > thresh;

% Show results
figure('Name','Image Edges','NumberTitle','off');
subplot(1,3,1), imshow(I), title('Original')
subplot(1,3,2), imshow(Igray), title('Grayscale')
subplot(1,3,3), imshow(E), title('Sobel edges')


