close all
addpath('../MatlabFunctions/');

% Set up wavelet decomposition parameters

J = 12;   % power of two used for selecting data points
L=1;
 
ismean = 1;          % measure used to compute wavelet spectra; options 0- mean/ 1 - median
isplot = 1;          % plot wavelet spectra 0 - No/ 1 - yes

filt=[ -0.075765714789341  -0.029635527645954 ...
   0.497618667632458   0.803738751805216 ...
   0.297857795605542  -0.099219543576935 ...
  -0.012603967262261   0.032223100604071]; %S ymmlet 


w = 2^9;  % window size
k1 = 5; k2 = log2(w)-1;  % range of scale used to compute wavelet spectra

% fix the number of features(windows) extracted from the window-based approach
max_row = 6900; min_row = 5900; num_windows = floor(min_row/w);

s = [ "Pachon", "Molino", "Surface" ];
for si = 1:length(s)
    % (1) Read Raw Data files into Matlab
    dirName = sprintf('../FishData/NewData/%s',s(si));  % folder path
    files = dir( fullfile(dirName,'*.mat') );  % list all *.mat files in a directory
    files = {files.name}';  %'# file names
    nfi = numel(files);

    % data matrix. row=num of fishes in a fish type * num of sub-signal
    window_rows = zeros(num_windows*nfi, max_col);
    for f_ind = 1:nfi
        fname = fullfile(dirName,files{f_ind});  %# full path to an individual data file
        data = load(fname);
        ca2_intensity =  data.deltaFoF;  % ca2+ intensity

        % (2) Compute a slope of wavelet coefficients of a windowed signal from each neuron
        % An individual neuron corresponds to a column of ca2_intensity
        [M, N] = size(ca2_intensity);
        for col_idx = 1:N
            x = ca2_intensity(:,col_idx);  % signal in one neuron
            x = x(~isnan(x));  % drop missing values
            
            % split data into windows of size w 
            windowed_x = SplitData(x, w);

            % compute entropy for each window
            window_slope = [];  % will store slope values of each window per neuron
            for j = 1 : num_windows
                % Perform Wavelet decomposition and compute slope
                [slope, levels, log2spec ] = waveletspectra(windowed_x{j}, L, filt, k1, k2, ismean, isplot);

                % window slopes stacked in a row (as a sample)
                window_rows((f_ind-1)*num_windows + j, col_idx) = slope;
            end
        end
    end
    % (3) Save a vector of entropies of a fish type into a csv file
    writematrix(window_rows, sprintf('../features/window512_neuron_slope_rowwindow_%s.%s',s(si),'csv'));
end