close all
addpath('../MatlabFunctions/');

% Set up wavelet decomposition parameters
%L=10;  % starting scale index of wavelet energy for entropy computation
L = 1;  % starting scale index of wavelet energy for entropy computation
ismean = 1;          % measure used to compute wavelet spectra; options 0- mean/ 1 - median
isplot = 0;          % plot wavelet spectra 0 - No/ 1 - yes

filt=[ -0.075765714789341  -0.029635527645954 ...
   0.497618667632458   0.803738751805216 ...
   0.297857795605542  -0.099219543576935 ...
  -0.012603967262261   0.032223100604071]; %S ymmlet 

J = 12;   % power of two used for selecting data points
w = 2^9;  % window size

% fix the number of features(windows) extracted from the window-based approach
max_row = 6900; min_row = 5900; num_windows = floor(min_row/w);
max_col = 879;  % at most, 879 neurons for a fish

s = [ "Pachon", "Molino", "Surface" ];
for si = 1 :length(s)
    % (1) Read Raw Data files into Matlab
    dirName = sprintf('../FishData/NewData/%s',s(si));  % folder path
    files = dir( fullfile(dirName,'*.mat') );  % list all *.mat files in a directory
    files = {files.name}';  %'# file names
    nfi = numel(files)
    % data matrix. row=num of fishes in a fish type * num of sub-signal
    window_rows = zeros(num_windows*nfi, max_col);
    for f_ind = 1:nfi
        fname = fullfile(dirName,files{f_ind});  %# full path to an individual data file
        data = load(fname); 
        ca2_intensity =  data.deltaFoF;  % ca2+ intensity

        % (2) Compute a shannon entropy of a windowed signal from each neuron
        % An individual neuron corresponds to a column of ca2_intensity
        [M, N] = size(ca2_intensity);
        for col_idx = 1:N
            x = ca2_intensity(:,col_idx);  % signal in one neuron
            x = x(~isnan(x));  % drop missing values
            
            % split data into windows of size w 
            windowed_x = SplitData(x, w);

            % compute entropy for each window
            window_ent = [];  % will store entropy values of each window per neuron 
            for j = 1 : num_windows
                %w_coeffs = dwtr(windowed_x{j}, J-L, filt);
                %wt_coeff = w_coeffs(2^(L)+1 : 2^(J));  % from L level to J level
                %debug_a = wt_coeff(wt_coeff==0);  % check whether any 0 in coefficients
                %if size(debug_a,1) >= 1 & size(debug_a,2) >= 1
                %    debug_a
                %end
                %Energy = wt_coeff.^2;
                %P = Energy/sum(Energy);  % normalize
                %ent = -sum(P.*log2(P))


                ent = wavelet_entropy(windowed_x{j}, filt, L)

                % window entropies stacked in a row (as a sample)
                %window_rows(j, col_idx) = ent;
                window_rows((f_ind-1)*num_windows + j, col_idx) = ent;
            end
        end
    end
    % (3) Save a vector of entropies of a fish type into a csv file
    writematrix(window_rows, sprintf('../features/window512_neuron_entropy_rowwindow_%s.%s',s(si),'csv'));
end