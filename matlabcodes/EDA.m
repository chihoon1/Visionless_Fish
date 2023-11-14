s = [ "Pachon", "Molino", "Surface" ];
min_M = 10000;  % store the minimum number of rows in ca2+ intensity of a subject(fish)
min_N = 10000;  % store the minimum number of columns in ca2+ intensity of a subject(fish)
max_N = 0;  % store the minimum number of columns in ca2+ intensity of a subject(fish)
for si = 1:length(s)
    % Read Raw Data files into Matlab
    dirName = sprintf('../FishData/NewData/%s',s(si));  % folder path
    files = dir( fullfile(dirName,'*.mat') );  %# list all *.xyz files in a directory
    files = {files.name}';  %'# file names
    nfi = numel(files);
    ent_class = [];
    for f_ind = 1:nfi
        fname = fullfile(dirName,files{f_ind});  %# full path to an individual data file
        data = load(fname); 
        ca2_intensity =  data.deltaFoF;  % ca2+ intensity
        [M, N] = size(ca2_intensity);
        if min_M > M
            min_M = M;
        end
        if min_N > N
            min_N = N;
        end
        if max_N < N
            max_N = N
        end
        for col_idx = 1:N
            x = ca2_intensity(:,col_idx);  % signal in one neuron
            % Check if missing value exists in a signal
            isnan_x = isnan(x);  % array of Boolean where i-th elem = 1 if xi == NaN
            if size(find(isnan_x == 1),1) > 0
                s(si); f_ind; col_idx;
                size(find(isnan_x == 1));
            end
        end
    end
end

%% Case group-1 : eyeless cavefish
dirName = sprintf('../FishData/NewData/Pachon');             %# folder path
files = dir( fullfile(dirName,'*.mat') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
nfi = numel(files);

i  = 3; 
fname = fullfile(dirName,files{i});     %# full path to file
data_p = load(fname); 
%% Case group-2  Molino : eyeless cavefish
dirName = sprintf('../FishData/NewData/Molino');             %# folder path
files = dir( fullfile(dirName,'*.mat') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
nfi = numel(files);

i  = 4; 
fname = fullfile(dirName,files{i});     %# full path to file
data_m = load(fname); 

%% Control group  Surface : eyed and river dweling fish
dirName = sprintf('../FishData/NewData/Surface');             %# folder path
files = dir( fullfile(dirName,'*.mat') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
nfi = numel(files);

i  = 7; 
fname = fullfile(dirName,files{i});     %# full path to file
data_s = load(fname);


%% Sample data signals (ca2+ intensity)

col_idx = 1;
figure(1)
subplot(311); plot(data_p.deltaFoF(:,col_idx)); title('Pachon'); grid on 
subplot(312); plot(data_m.deltaFoF(:,col_idx)); title('Molino'); grid on 
ylabel('Average Ca^{2+} Intensity')
subplot(313); plot(data_s.deltaFoF(:,col_idx)); title('Surface'); grid on
xlabel('Time')

%% Sample data signals (ImageAverage)

col_idx = 1;
figure(2)
subplot(311); plot(data_p.imageAvg(:,col_idx)); title('Pachon'); grid on 
subplot(312); plot(data_m.imageAvg(:,col_idx)); title('Molino'); grid on 
ylabel('Image Average')
subplot(313); plot(data_s.imageAvg(:,col_idx)); title('Surface'); grid on
xlabel('Time')

%% Sample data signals (F0)

col_idx = 1;
figure(3)
subplot(311); plot(data_p.F0(:,col_idx)); title('Pachon'); grid on 
subplot(312); plot(data_m.F0(:,col_idx)); title('Molino'); grid on 
ylabel('F0')
subplot(313); plot(data_s.F0(:,col_idx)); title('Surface'); grid on
xlabel('Time')

%% Sample data signals (movements)

figure(4)
subplot(311); plot(data_p.movements); title('Pachon'); grid on 
subplot(312); plot(data_m.movements); title('Molino'); grid on 
ylabel('movements')
subplot(313); plot(data_s.movements); title('Surface'); grid on
xlabel('Time')

%% Sample data signals (raster)

col_idx = 1;
figure(5)
subplot(311); plot(data_p.raster(:,col_idx)); title('Pachon'); grid on 
subplot(312); plot(data_m.raster(:,col_idx)); title('Molino'); grid on 
ylabel('raster')
subplot(313); plot(data_s.raster(:,col_idx)); title('Surface'); grid on
xlabel('Time')


min_M
min_N
max_N

