%% ========================================================================
%  AETHER-P3 Dataset Builder (Seq2Seq): Inputs + Targets + Empirical Baselines
%
%  Purpose:
%   Build the training/evaluation tensors used by AETHER-P3 for thermospheric
%   density forecasting:
%     - x_hist: historical conditioning sequence (length m)
%     - x_fut : future query sequence (length n) specifying where/when to forecast
%     - ydata : target density sequence (length n)
%
%  AETHER-P3 framing:
%   - Sequence-to-sequence regression:
%       Input  : last m steps of space-weather drivers + baseline densities
%       Query  : the future times/locations (n steps) to be predicted
%       Output : density at those future times/locations (n steps)
%   - Physics-informed conditioning:
%       JB2008 + NRLMSIS-00 densities are evaluated at *future query points*
%       and appended to the historical features, so AETHER-P3 learns residual/
%       corrections and generalizes across conditions.
%
%  Data domains:
%   - ydata stored in log10(density)
%   - Empirical baseline outputs are also stored in log10(density)
%
%  Files / dependencies:
%   - Satellite raw files selected via uigetdir (CDF for CHAMP/GRACE/SWARM, TXT for GOCE)
%   - Solar / geomagnetic / solar-wind indices loaded from local datafiles
%   - Modified JB2008 and NRLMSISE-00 packages are required in Packages/
%
%  Outputs:
%   - Output/raw_data.mat       : downsampled raw satellite data table (save_data)
%   - Output/ydata.mat          : ydata  [Nsamples x n]
%   - Output/x_hist_data.mat    : x_hist [Nsamples x m x (9 + 2*n)]
%   - Output/x_fut_data.mat     : x_fut  [Nsamples x n x 9]
%
%  Notes:
%   - "interval" assumes raw cadence is 10 seconds (downsample step = interval/10).
%   - Samples with invalid / non-finite entries are removed before saving.
% ========================================================================

clear; clc; close all

%% =========================
%  Initial settings
%  interval : downsample interval in seconds (target resolution)
%  m        : number of historical steps (input sequence length)
%  n        : number of future steps (forecast horizon length)
%  sat      : satellite selection (raw-data format differs per mission)
%% =========================
interval = 10*60; % resolution (seconds)
m = 18;           % number of the input sequences
n = 36;           % number of the output sequences

sat = 3; % 1:CHAMP 2:GRACE 3:SWARM-C 4:GOCE

%% =========================
%  Add package paths
%  - spdfcdfread (CDF reader)
%  - Modified JB2008 model implementation
%  - Modified NRLMSISE-00 model implementation
%% =========================
addpath Packages/matlab_cdf390_patch-64/
addpath Packages/Modified_JB2008/
addpath Packages/Modified_NRLMSISE00/

%% =========================
%  Load indices data files (space weather drivers)
%  Used by JB2008 / MSIS and by AETHER-P3 historical conditioning features.
%% =========================
eopdata  = readmatrix('EOP-ALL.txt')';
DTCdata  = readmatrix('DTCFILE.TXT')';
SOLdata  = readmatrix('SOLFSMY.TXT')';
swdata   = readmatrix('SW-All.txt')';
Fluxdata = readmatrix('Datafiles/radio_flux_adjusted.txt')';

load('Datafiles/Apo30.mat');      % provides apo_data
load('Datafiles/DST.mat')         % provides DSTdata
load('Datafiles/solarwind00_10.mat');
SolarWind = data;                 % rename to a consistent variable

%% =========================
%  Initialize empirical models (used as AETHER-P3 conditioners)
%  JB2008 requires JPL ephemeris coefficients and global constants.
%  MSIS requires flags/input structs (reused for speed).
%% =========================

% --- JB2008 Model ---
global PC const
run(fullfile('Packages','Modified_JB2008','SAT_Const.m'));
run(fullfile('Packages','Modified_JB2008','constants.m'));
Coef = load('Packages/Modified_JB2008/DE430Coeff.mat');
PC = Coef.DE430Coeff;
wgs84 = wgs84Ellipsoid('meters'); % geodetic <-> ECEF conversions

% --- NRLMSIS-00 Model ---
flags = struct('switches',[0; ones(23,1)],'sw',zeros(24,1),'swc',zeros(24,1));
flags.switches(10)=-1;
ap = struct('a',zeros(8,1));
input = struct('year',0,'doy',0,'sec',0,'alt',0,'g_lat',0,'g_long',0,'lst',0,'f107A',0,'f107',0,'ap',0,'ap_a',ap);

%% =========================
%  Read and preprocess raw satellite data
%  - User selects a folder containing raw files.
%  - CHAMP/GRACE/SWARM: *.cdf
%  - GOCE            : *.txt
%  Output "save_data" is downsampled and saved as Output/raw_data.mat
%% =========================
save_data = [];

folder = uigetdir;
switch sat
    case {1, 2, 3}
        filepath = dir(fullfile(folder, '*.cdf'));
        for i = 1:length(filepath)
            filename = filepath(i).name
            row_data = spdfcdfread([folder, '/', filename]);

            % UTC time (Y M D h m s)
            time = datevec(row_data{1});

            % Save format expected by downstream processing:
            % [time(1:6), density, alt(m), lat(deg), lon(deg), lst(h)]
            data = [time, row_data{2}, row_data{3}, row_data{4}, row_data{5}, row_data{6}];
            save_data = [save_data; data(:,1:end)];
        end

    case 4
        filepath = dir(fullfile(folder, '*.txt'));
        for i = 1:length(filepath)
            filename = filepath(i).name
            row_data = readmatrix([folder, '/', filename]);
            data = readtable([folder, '/', filename]);

            % Parse date/time strings in GOCE text files
            ymd = str2double(split(string(data.Var1), '-'));
            hms = str2double(split(string(data.Var2), ':'));
            year   = ymd(:, 1);
            month  = ymd(:, 2);
            days   = ymd(:, 3);
            hour   = hms(:, 1);
            minute = hms(:, 2);
            second = hms(:, 3);

            % GOCE column selection preserved from your code:
            % [Y M D h m s, alt, lon, lat, lst, density] (based on row_data mapping)
            month_data = [year, month, days, hour, minute, second, row_data(:, [4:7, 9])];
            save_data = [save_data; month_data];
        end
end

% Downsample: assumes raw cadence is 10 s (hence interval/10 step)
save_data = save_data(1:interval/10:end, :);
save('Output/raw_data.mat', 'save_data');

%% =========================
%  Unpack columns into standard variables
%  Different satellites store density/alt/lat/lon/lst in different columns.
%% =========================
years   = save_data(:,1);
months  = save_data(:,2);
days    = save_data(:,3);
hours   = save_data(:,4);
minutes = save_data(:,5);
seconds = save_data(:,6);

switch sat
    case {1, 2}
        alt     = save_data(:, 7);
        lat     = save_data(:, 8);
        lon     = save_data(:, 9);
        lst     = save_data(:, 10);
        density = save_data(:, 11);
    case 3
        density = save_data(:, 7);
        alt     = save_data(:, 8);
        lat     = save_data(:, 9);
        lon     = save_data(:, 10);
        lst     = save_data(:, 11);
    case 4
        alt     = save_data(:, 7);
        lon     = save_data(:, 8);
        lat     = save_data(:, 9);
        lst     = save_data(:, 10);
        density = save_data(:, 11);
end

T = length(density);

% -------------------- Allocate AETHER-P3 tensors -------------------------
% ydata  : log10 density targets for horizons 1..n
% x_hist : historical drivers (m steps) + future-baseline densities (2*n)
% x_fut  : future query points (n steps) = [lat_rad lon_rad alt_km + time encodings]
ydata  = nan(T, n);
x_hist = nan(T, m, 8+2*n);
x_fut  = nan(T, n, 9);

%% =========================
%  Generate inputs and outputs (seq2seq windows)
%  For each start index i:
%   - future target density window is density(i+m : i+m+n-1)
%   - x_hist_seq uses indices j = i ... i+m-1 (historical conditioning)
%   - x_fut_seq uses indices ii = i+m+p (future query points, p=0..n-1)
%
%  Important design choice for AETHER-P3:
%   - For each historical time j, we evaluate JB2008/MSIS at the future query
%     points (i+m+p). This provides "baseline density at requested future
%     locations," which AETHER-P3 can use as a physics-informed conditioner.
%% =========================
for i = 1:T-n-m
    i

    future_density = density(i+m:i+m+n-1);

    % Skip sample if future density is outside sanity bounds
    if any(future_density > 1 | future_density < 0)
        continue
    end

    x_hist_seq = nan(m, 8+2*n);
    x_fut_seq  = nan(n, 9);

    for j = i:i+m-1
        % Current (historical) time bookkeeping
        doy = finddays(years(j),months(j),days(j),hours(j),minutes(j),seconds(j));
        MJD = Mjday(years(j),months(j),days(j),hours(j),minutes(j),seconds(j));

        % rho_pred concatenates [log10(JB at future_1), log10(MSIS at future_1), ...] for p=0..n-1
        rho_pred = [];

        for p = 0:n-1
            % ----- Future query point index -----
            % Use future location at ii = i+m+p
            [X,Y,Z] = geodetic2ecef(wgs84,lat(i+m+p),lon(i+m+p),alt(i+m+p),'degrees');
            ecef = [X; Y; Z];
            llh  = ecf2llhT(ecef);      % [lat(rad), lon(rad), h(m)]
            lat_rad = llh(1);
            lon_rad = llh(2);
            alt_km  = llh(3)/1000;

            % ----- Future time encodings (query features) -----
            future_time = datetime([years(i+m+p), months(i+m+p), days(i+m+p), hours(i+m+p), minutes(i+m+p), seconds(i+m+p)]);
            future_doy  = day(future_time, 'dayofyear');
            future_hour = hours(i+m+p);
            future_lst  = lst(i+m+p);

            future_t1 = sin(2*pi*future_doy/365.25);
            future_t2 = cos(2*pi*future_doy/365.25);
            future_t3 = sin(2*pi*future_hour / 24);
            future_t4 = cos(2*pi*future_hour / 24);
            future_t5 = sin(2*pi*future_lst /24);
            future_t6 = cos(2*pi*future_lst /24);

            % Future query tensor (AETHER-P3 query conditioning)
            x_fut_seq(p+1, :) = [lat_rad, lon_rad, alt_km, future_t1, future_t2, future_t3, future_t4, future_t5, future_t6];

            % ----- Empirical baselines evaluated at the future query point -----
            % JB2008 is driven by time j (historical context) but evaluated at future location.
            EstiJB = JB2008model(years(j),months(j),days(j),hours(j),minutes(j),seconds(j), doy, MJD, lat_rad, lon_rad, alt_km, SOLdata, DTCdata, eopdata);

            % NRLMSIS is also driven by historical time j but evaluated at future location.
            EstiMSIS = NRLMSIS00model(years(j),months(j),days(j),hours(j),minutes(j),seconds(j), lat(i+m+p),lon(i+m+p),alt_km, lst(j), SOLdata, swdata, flags, input);

            rho_pred = [rho_pred, log10(EstiJB), log10(EstiMSIS)];
        end

        % Historical drivers at time j (space weather + solar wind + time encodings)
        [Dst, F107, F107A, F30, Apo, t1, t2, t3, t4, t5, t6, Bz, Speed, Proton, AE] = ...
            get_indices(years(j),months(j),days(j),hours(j),minutes(j),seconds(j), lst(j), SOLdata, Fluxdata, apo_data, DSTdata, SolarWind);

        % Historical feature vector (per time step j):
        % [solar fluxes + geomag + solar wind + baseline predictions]
        x_hist_seq(j-i+1, :) = [F107, F107A, F30, Dst, Apo, Bz, Speed, Proton, AE, rho_pred];
    end

    % Store this sample
    x_fut(i, :, :)  = x_fut_seq;
    ydata(i, :)     = log10(future_density)'; % target in log10 domain
    x_hist(i, :, :) = x_hist_seq;
end

%% =========================
%  Remove samples with NaN / Inf (data hygiene)
%  Any sample with non-finite entries in x_hist, x_fut, or ydata is discarded.
%% =========================
S = size(x_hist, 1);

hist_bad = any(~isfinite(reshape(x_hist, S, [])), 2);
fut_bad  = any(~isfinite(reshape(x_fut , S, [])), 2);
y_bad    = any(~isfinite(ydata), 2);

bad  = hist_bad | fut_bad | y_bad;
keep = ~bad;

x_hist = x_hist(keep, :, :);
x_fut  = x_fut (keep, :, :);
ydata  = ydata(keep, :);

fprintf('Cleaned: kept %d / %d samples (removed %d with NaN/Inf)\n', ...
    sum(keep), S, sum(bad));

%% =========================
%  Save tensors for AETHER-P3 training/inference
%% =========================
save('Output/ydata.mat', 'ydata', '-v7.3');
save('Output/x_hist_data.mat', 'x_hist', '-v7.3')
save('Output/x_fut_data.mat', 'x_fut', '-v7.3')

%% ========================================================================
%  Functions
%  JB2008model      : wrapper returning density at the query location/altitude
%  NRLMSIS00model   : wrapper returning density at the query location/altitude
%  get_indices      : returns space-weather/solar-wind drivers at historical time
%% ========================================================================

function RHO = JB2008model(years, months, days, hours, minutes, seconds, doy, MJD, lat, lon, alt, SOLdata, DTCdata, eopdata)
global const PC

% read matched solar indices
% use 1 day lag for F10 and S10 (JB2008)
i = find(years==SOLdata(1,:) & floor(doy)==SOLdata(2,:));
SOL = SOLdata(:, i-1);
F10 = SOL(4);
F10B = SOL(5);
S10 = SOL(6);
S10B = SOL(7);

% use 2 days lag for M10 (JB2008)
SOL = SOLdata(:, i-2);
XM10 = SOL(8);
XM10B = SOL(9);

% use 5 days lag for Y10 (JB2008)
SOL = SOLdata(:, i-5);
Y10 = SOL(10);
Y10B = SOL(11);

% SELECT GEOMAGNETIC DTC VALUE
i = years==DTCdata(1,:) & floor(doy)==DTCdata(2,:);
DTC = DTCdata(:, i);
DSTDTC = DTC(floor(hours)+3);

% CONVERT LONGITUDE TO RA using Earth orientation / time standards
[x_pole,y_pole,UT1_UTC,LOD,dpsi,deps,dx_pole,dy_pole,TAI_UTC] = IERS_JB(eopdata,MJD,'l');
[UT1_TAI,UTC_GPS,UT1_GPS,TT_UTC,GPS_UTC] = timediff(UT1_UTC,TAI_UTC);
[DJMJD0, DATE] = iauCal2jd(years, months, days);
TIME = (60*(60*hours+minutes)+seconds)/86400;
UTC = DATE+TIME;
TT = UTC+TT_UTC/86400;
TUT = TIME+UT1_UTC/86400;
UT1 = DATE+TUT;
GWRAS = iauGmst06(DJMJD0, UT1, DJMJD0, TT);

% SAT state in the JB2008 expected frame/format:
% [RA-longitude, latitude, altitude(km)]
XLON = lon;
SAT(1) = mod(GWRAS + XLON, 2*pi);
SAT(2) = lat;
SAT(3) = alt;

% Sun right ascension/declination from JPL ephemeris
MJD_TDB = Mjday_TDB(TT);
[r_Mercury,r_Venus,r_Earth,r_Mars,r_Jupiter,r_Saturn,r_Uranus, ...
    r_Neptune,r_Pluto,r_Moon,r_Sun,r_SunSSB] = JPL_Eph_DE430(MJD_TDB);
ra_Sun  = atan2(r_Sun(2), r_Sun(1));
dec_Sun = atan2(r_Sun(3), sqrt(r_Sun(1)^2+r_Sun(2)^2));
SUN(1)  = ra_Sun;
SUN(2)  = dec_Sun;

[TEMP,RHO] = JB2008(MJD,SUN,SAT,F10,F10B,S10,S10B,XM10,XM10B,Y10,Y10B,DSTDTC); %#ok<ASGLU>
end

function EstiMSIS = NRLMSIS00model(years, months, days, hours, minutes, seconds, lat, lon, alt, lst, SOLdata, swdata, flags, input)
% Build day-of-year and populate MSIS input struct
doy = floor(finddays (years, months, days, hours, minutes, seconds));

input.year   = years; % kept as-is (original code notes "without effect")
input.doy    = doy;
input.sec    = hours*3600+minutes*60+seconds;
input.alt    = alt;
input.g_lat  = lat;
input.g_long = lon;
input.lst    = lst;   % local solar time from satellite (preferred)

% Match swdata row (year, month, day)
i = years==swdata(1,:) & months==swdata(2,:) & days==swdata(3,:);
sw = swdata(:, i);

% Insert F10.7 and F10.7A from SOLdata into sw array (kept as-is)
ii = find(years==SOLdata(1,:) & doy==SOLdata(2,:));
sw(31) = SOLdata(4,ii);
sw(32) = SOLdata(5,ii);
sw(33) = sw(32);

% Define 3-hour ap bins based on UT hour
if hours < 3
    input.ap_a.a(1) = sw(15);
elseif hours < 6
    input.ap_a.a(2) = sw(16);
elseif hours < 9
    input.ap_a.a(3) = sw(17);
elseif hours < 12
    input.ap_a.a(4) = sw(18);
elseif hours < 15
    input.ap_a.a(5) = sw(19);
elseif hours < 18
    input.ap_a.a(6) = sw(20);
elseif hours < 21
    input.ap_a.a(7) = sw(21);
else
    input.ap_a.a(8) = sw(22);
end

input.ap   = sw(23);
input.f107 = sw(31);
input.f107A = sw(32);

EstiMSIS = nrlmsise00(input,flags);
end

function [Dst, F107, F107A, F30, Apo, t1, t2, t3, t4, t5, t6, Bz, Speed, Proton, AE] = ...
    get_indices(years, months, days, hours, minutes, seconds, lst, SOLdata, Fluxdata, apo_data, DSTdata, SolarWind)

local_time = datetime([years, months, days, hours, minutes, seconds]);
doy = day(local_time, 'dayofyear');

% Dst index (hourly)
i = years==DSTdata(:, 1) & months==DSTdata(:, 2) & days==DSTdata(:, 3) & hours==DSTdata(:, 4);
Dst = DSTdata(i, 8);

% F10.7 and F10.7A (SOLdata)
j = years==SOLdata(1,:) & floor(doy)==SOLdata(2,:);
SOL = SOLdata(:, j);
F107  = SOL(4);
F107A = SOL(5);

% F30 (radio flux file)
k = years==Fluxdata(1,:) & months==Fluxdata(2,:) & days==Fluxdata(3,:);
F30 = Fluxdata(5,k);

% Apo30 (30-minute indexing logic preserved)
l = find(years==apo_data(:, 1) & months==apo_data(:, 2) & days==apo_data(:, 3) & hours==apo_data(:, 4));
if minutes < 30
    ll = l;
else
    ll = l+1;
end
Apo = apo_data(ll, 9);

% time encodings (useful periodic features for AETHER-P3)
t1 = sin(2*pi*doy/365.25);
t2 = cos(2*pi*doy/365.25);
t3 = sin(2*pi*hours / 24);
t4 = cos(2*pi*hours / 24);
t5 = sin(2*pi*lst /24);
t6 = cos(2*pi*lst /24);

% Solar wind features (minute-level lookup)
m = find(years==SolarWind(:, 1) & doy==SolarWind(:, 2) & hours==SolarWind(:, 3) & minutes==SolarWind(:, 4));
Bz     = SolarWind(m, 8);
Speed  = SolarWind(m, 9);
Proton = SolarWind(m, 13);
AE     = SolarWind(m, 14);

end
