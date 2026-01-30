%% ========================================================================
%  Satellite Altitude Coverage + Solar Activity Context (F10.7A)
%
%  Purpose:
%   This script visualizes the altitude-time coverage of multiple satellite
%   datasets (CHAMP, GRACE-A, GOCE, SWARM-C) and the altitude ranges of your
%   designated test cases, together with the 81-day mean solar flux F10.7A.
% ========================================================================

clear; clc; close all; format long

n = 36; % Forecast horizon length (steps); used for windowing/alignment

%% =========================
%  Load CHAMP dataset
%  - raw_data.mat is expected to contain save_data
%  - For CHAMP, density is assumed at column 11, altitude at column 7
%% =========================
load("Output/Data_3hr_6hr/Champ00_10/raw_data.mat");
m = numel(save_data(:, 11));                % number of samples (based on density column)
den_Champ = save_data(:, 11);               % density series
den_Champ_valid = den_Champ((0:n-1)+(1:(m-n+1))');  % windowed density (each row is one n-step window)

time_Champ = datetime(save_data(:, 1:3));   % date axis (Y-M-D)
time_Champ_valid = time_Champ((0:n-1)+(1:(m-n+1))'); % windowed time stamps

% Keep only windows where all n densities satisfy a sanity constraint (< 1)
index = find(all(den_Champ_valid<1, 2));

% Map the selected windows back to altitude/time series for plotting
alt_Champ = save_data(index, 7)/1000;       % altitude in km
date_Champ = time_Champ_valid(index, 1);    % representative time stamp per window
alt_Champ_mean = movmean(alt_Champ, [0, 500]); % long smoothing to show trend

%% =========================
%  Load GRACE-A dataset
%  Same conventions as CHAMP: density col 11, altitude col 7
%% =========================
load("Output/Data_3hr_6hr/GRACE-A_02_17/raw_data.mat");
m = numel(save_data(:, 11));
den_Grace = save_data(:, 11);
den_Grace_valid = den_Grace((0:n-1)+(1:(m-n+1))');

time_Grace = datetime(save_data(:, 1:3));
time_Grace_valid = time_Grace((0:n-1)+(1:(m-n+1))');

index = find(all(den_Grace_valid<1, 2));
alt_Grace = save_data(index, 7)/1000;
date_Grace = time_Grace_valid(index, 1);
alt_Grace_mean = movmean(alt_Grace, [0, 500]);

%% =========================
%  Load GOCE dataset
%  Same conventions as CHAMP/GRACE: density col 11, altitude col 7
%% =========================
load("Output/Data_3hr_6hr/GOCE_09_13/raw_data.mat");
m = numel(save_data(:, 11));
den_Goce = save_data(:, 11);
den_Goce_valid = den_Goce((0:n-1)+(1:(m-n+1))');

time_Goce = datetime(save_data(:, 1:3));
time_Goce_valid = time_Goce((0:n-1)+(1:(m-n+1))');

index = find(all(den_Goce_valid<1, 2));
alt_Goce = save_data(index, 7)/1000;
date_Goce = time_Goce_valid(index, 1);
alt_Goce_mean = movmean(alt_Goce, [0, 500]);

%% =========================
%  Load SWARM-C dataset
%  SWARM format differs:
%   - density is at column 7
%   - altitude is at column 8
%% =========================
load("Output/Data_3hr_6hr/SWARM-C_14_19/raw_data.mat");
m = numel(save_data(:, 7));
den_Swarm = save_data(:, 7);
den_Swarm_valid = den_Swarm((0:n-1)+(1:(m-n+1))');

time_Swarm = datetime(save_data(:, 1:3));
time_Swarm_valid = time_Swarm((0:n-1)+(1:(m-n+1))');

index = find(all(den_Swarm_valid<1, 2));
alt_Swarm = save_data(index, 8)/1000;
date_Swarm = time_Swarm_valid(index, 1);
alt_Swarm_mean = movmean(alt_Swarm, [0, 500]);

%% =========================
%  Load test-case raw data (altitude tracks)
%  These are the specific evaluation segments.
%% =========================
load("Output/Model/Model1/Test/Test1/raw_data.mat")
alt_Test1 = save_data(:, 8)/1000;
date_Test1 = datetime(save_data(:, 1:3));
alt_Test1_mean = movmean(alt_Test1, [0, 10]);

load("Output/Model/Model1/Test/Test2/raw_data.mat")
alt_Test2 = save_data(:, 8)/1000;
date_Test2 = datetime(save_data(:, 1:3));
alt_Test2_mean = movmean(alt_Test2, [0, 10]);

load("Output/Model/Model1/Test/Test3/raw_data.mat")
alt_Test3 = save_data(:, 8)/1000;
date_Test3 = datetime(save_data(:, 1:3));
alt_Test3_mean = movmean(alt_Test3, [0, 10]);

load("Output/Model/Model1/Test/Test4/raw_data.mat")
alt_Test4 = save_data(:, 8)/1000;
date_Test4 = datetime(save_data(:, 1:3));
alt_Test4_mean = movmean(alt_Test4, [0, 10]);

load("Output/Model/Model1/Test/Test5/raw_data.mat")
alt_Test5 = save_data(:, 7)/1000;
date_Test5 = datetime(save_data(:, 1:3));
alt_Test5_mean = movmean(alt_Test5, [0, 10]);

load("Output/Model/Model1/Test/Test6/raw_data.mat")
alt_Test6 = save_data(:, 8)/1000;
date_Test6 = datetime(save_data(:, 1:3));
alt_Test6_mean = movmean(alt_Test6, [0, 10]);

load("Output/Model/Model1/Test/Test7/raw_data.mat")
alt_Test7 = save_data(:, 8)/1000;
date_Test7 = datetime(save_data(:, 1:3));
alt_Test7_mean = movmean(alt_Test7, [0, 10]);

load("Output/Model/Model1/Test/Test8/raw_data.mat")
alt_Test8 = save_data(:, 8)/1000;
date_Test8 = datetime(save_data(:, 1:3));
alt_Test8_mean = movmean(alt_Test8, [0, 10]);

%% =========================
%  Load solar flux (F10.7A) and build datetime axis
%  - SOLdata format assumed: year row, DOY row, F10.7 row, F10.7A row
%% =========================
SOLdata = readmatrix('Datafiles/SOLDATA.TXT')';
SOL_year_1 = SOLdata(1, 1:7061)';
SOL_DOY_1  = SOLdata(2, 1:7061)';
F107_1     = SOLdata(4, 1:7061)';    %#ok<NASGU> % loaded for completeness
F107A_1    = SOLdata(5, 1:7061)';

SOL_year_2 = SOLdata(1, 8557:8922)';
SOL_DOY_2  = SOLdata(2, 8557:8922)';
F107_2     = SOLdata(4, 8557:8922)'; %#ok<NASGU>
F107A_2    = SOLdata(5, 8557:8922)';

SOL_Date_1 = datetime(SOL_year_1, 1, 1) + days(SOL_DOY_1-1);
SOL_Date_2 = datetime(SOL_year_2, 1, 1) + days(SOL_DOY_2-1);

%% =========================
%  Plot
%  Left axis : altitude (km) of each satellite/test track
%  Right axis: F10.7A (81-day mean) as solar-cycle context
%% =========================
figure(1)

yyaxis left
hold on

% --- CHAMP ---
hline1 = plot(date_Champ, alt_Champ, '-');
hline1.Color=[102/255, 255/255, 153/255, 0.1];
hline2 = plot(date_Champ, alt_Champ_mean,'-', 'LineWidth',2);
hline2.Color=[0/255, 204/255, 0/255, 1];

% --- GRACE-A ---
hline3 = plot(date_Grace, alt_Grace, '-');
hline3.Color=[102/255, 204/255, 255/255, 0.1];
hline4 = plot(date_Grace, alt_Grace_mean,'-', 'LineWidth',2);
hline4.Color=[0/255, 153/255, 255/255, 1];

% --- GOCE ---
hline5 = plot(date_Goce, alt_Goce, '-');
hline5.Color=[255/255, 153/255, 255/255, 0.1];
hline6 = plot(date_Goce, alt_Goce_mean,'-', 'LineWidth',2);
hline6.Color=[204/255, 0/255, 204/255, 1];

% --- SWARM-C ---
hline7 = plot(date_Swarm, alt_Swarm, '-');
hline7.Color=[255/255, 153/255, 153/255, 0.1];
hline8 = plot(date_Swarm, alt_Swarm_mean,'-', 'LineWidth',2);
hline8.Color=[204/255, 122/255, 122/255, 1];

% --- Test Cases (same styling across tests) ---
hline9  = plot(date_Test1, alt_Test1, '-');        hline9.Color =[232/255, 224/255, 239/255, 0.1];
hline10 = plot(date_Test1, alt_Test1_mean,'-', 'LineWidth',1); hline10.Color=[194/255, 177/255, 215/255, 1];

hline11 = plot(date_Test2, alt_Test2, '-');        hline11.Color=[232/255, 224/255, 239/255, 0.1];
hline12 = plot(date_Test2, alt_Test2_mean,'-', 'LineWidth',1); hline12.Color=[194/255, 177/255, 215/255, 1];

hline13 = plot(date_Test3, alt_Test3, '-');        hline13.Color=[232/255, 224/255, 239/255, 0.1];
hline14 = plot(date_Test3, alt_Test3_mean,'-', 'LineWidth',1); hline14.Color=[194/255, 177/255, 215/255, 1];

hline15 = plot(date_Test4, alt_Test4, '-');        hline15.Color=[232/255, 224/255, 239/255, 0.1];
hline16 = plot(date_Test4, alt_Test4_mean,'-', 'LineWidth',1); hline16.Color=[194/255, 177/255, 215/255, 1];

hline23 = plot(date_Test5, alt_Test5, '-');        hline23.Color=[232/255, 224/255, 239/255, 0.1];
hline24 = plot(date_Test5, alt_Test5_mean,'-', 'LineWidth',1); hline24.Color=[194/255, 177/255, 215/255, 1];

hline25 = plot(date_Test6, alt_Test6, '-');        hline25.Color=[232/255, 224/255, 239/255, 0.1];
hline26 = plot(date_Test6, alt_Test6_mean,'-', 'LineWidth',1); hline26.Color=[194/255, 177/255, 215/255, 1];

hline27 = plot(date_Test7, alt_Test7, '-');        hline27.Color=[232/255, 224/255, 239/255, 0.1];
hline28 = plot(date_Test7, alt_Test7_mean,'-', 'LineWidth',1); hline28.Color=[194/255, 177/255, 215/255, 1];

hline29 = plot(date_Test8, alt_Test8, '-');        hline29.Color=[232/255, 224/255, 239/255, 0.1];
hline30 = plot(date_Test8, alt_Test8_mean,'-', 'LineWidth',1); hline30.Color=[194/255, 177/255, 215/255, 1];

xlabel('Year')
ylabel('Altitude (km)')

yyaxis right

% --- F10.7A (81-day mean) ---
hline31 = plot(SOL_Date_1, F107A_1, '-', 'LineWidth',1.5);
hline31.Color=[178/255, 178/255, 178/255, 0.5];

hline32 = plot(SOL_Date_2, F107A_2, '-', 'LineWidth',1.5);
hline32.Color=[178/255, 178/255, 178/255, 0.5];

ylim([50, 250])
ylabel('81-day mean $F_{10.7}$ (s.f.u)','Interpreter','latex')

grid on
