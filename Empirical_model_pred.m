%% ========================================================================
%  Empirical Density Baselines Generation for AETHER-P3 (Inputs/Conditioners)
%
%  Purpose:
%   This script generates *future-location-conditioned* baseline thermospheric
%   density estimates from:
%     1) JB2008
%     2) NRLMSISE-00
%   at the *same future time/location grid* used by the learning-based model.
%
%  Outputs:
%   - Output/raw_data.mat     : downsampled satellite-derived samples
%   - Output/jb2008_pred.mat  : y_jb   [Nsamples x n] in log10 domain
%   - Output/msis_pred.mat    : y_msis [Nsamples x n] in log10 domain
%
% ========================================================================

clear; clc; close all

%% =========================
%  Settings
%  interval : target sampling interval for the raw satellite data (seconds)
%  m        : history length (steps). Used only to shift the forecast start.
%  n        : number of forecast horizons (steps)
%  sat      : satellite format selector
%             1: CHAMP   2: GRACE   3: SWARM
%% =========================
interval = 10*60;   % seconds
m = 18;             % history length (only for alignment)
n = 36;             % forecast horizons
sat = 3;            % 1:CHAMP 2:GRACE 3:SWARM

%% =========================
%  Paths
%  Add dependencies for:
%   - CDF reading (SPDF)
%   - Modified JB2008 implementation + its constants
%   - Modified NRLMSISE-00 implementation + wrapper
%% =========================
addpath Packages/matlab_cdf390_patch-64/
addpath Packages/Modified_JB2008/
addpath Packages/Modified_NRLMSISE00/

%% =========================
%  Load indices / aux files
%  These files provide:
%   - SOLdata : solar indices for JB2008 / MSIS (F10.7, F10.7A, etc.)
%   - DTCdata : DTC correction for JB2008
%   - eopdata : Earth orientation parameters for JB2008 timing/frames
%   - swdata  : space weather / Ap-related inputs for MSIS wrapper
%% =========================
eopdata  = readmatrix('EOP-ALL.txt')';
DTCdata  = readmatrix('DTCFILE.TXT')';
SOLdata  = readmatrix('SOLFSMY.TXT')';
swdata   = readmatrix('SW-All.txt')';

% (DST/Flux/etc not required for the JB/MSIS calls shown here)
% Fluxdata = readmatrix('Datafiles/radio_flux_adjusted.txt')';
% load('Datafiles/DST.mat')

%% =========================
%  JB2008 initialization
%  Initializes global constants and JPL ephemeris coefficients used by JB2008.
%% =========================
global PC const
run(fullfile('Packages','Modified_JB2008','SAT_Const.m'));
run(fullfile('Packages','Modified_JB2008','constants.m'));
Coef = load('Packages/Modified_JB2008/DE430Coeff.mat');
PC   = Coef.DE430Coeff;

wgs84 = wgs84Ellipsoid('meters');  % used for geodetic <-> ECEF conversions

%% =========================
%  NRLMSIS initialization
%  Pre-allocate flags and input structs for the MSIS wrapper.
%% =========================
flags = struct('switches',[0; ones(23,1)],'sw',zeros(24,1),'swc',zeros(24,1));
flags.switches(10) = -1;
ap    = struct('a',zeros(8,1));
input = struct('year',0,'doy',0,'sec',0,'alt',0,'g_lat',0,'g_long',0,'lst',0, ...
               'f107A',0,'f107',0,'ap',0,'ap_a',ap);

%% =========================
%  Read raw satellite data
%  - User selects a folder containing *.cdf files
%  - Each CDF file is read and concatenated into save_data
%  - save_data is then downsampled to "interval"
%  Columns after the first 6 time fields depend on the satellite type.
%% =========================
folder   = uigetdir;
save_data = [];

filepath = dir(fullfile(folder, '*.cdf'));
for k = 1:length(filepath)
    filename = filepath(k).name
    row_data = spdfcdfread(fullfile(folder, filename));
    time = datevec(row_data{1}); % [Y M D h m s]
    data = [time, row_data{2}, row_data{3}, row_data{4}, row_data{5}, row_data{6}];
    save_data = [save_data; data];
end

% downsample to desired resolution (CDF is assumed to be 10-second cadence)
save_data = save_data(1:interval/10:end, :);
if ~exist('Output', 'dir'); mkdir('Output'); end
save('Output/raw_data.mat', 'save_data');

%% =========================
%  Unpack columns (format depends on sat)
%  This block standardizes the variables used later:
%   years, months, days, hours, minutes, seconds
%   alt_m, lat_deg, lon_deg, lst_h, density
%% =========================
years   = save_data(:,1);
months  = save_data(:,2);
days    = save_data(:,3);
hours   = save_data(:,4);
minutes = save_data(:,5);
seconds = save_data(:,6);

switch sat
    case {1,2}
        alt_m   = save_data(:, 7);
        lat_deg = save_data(:, 8);
        lon_deg = save_data(:, 9);
        lst_h   = save_data(:,10);
        density = save_data(:,11);
    case 3
        density = save_data(:, 7);
        alt_m   = save_data(:, 8);
        lat_deg = save_data(:, 9);
        lon_deg = save_data(:,10);
        lst_h   = save_data(:,11);
end

T = length(density);

% outputs: future n-step baseline estimates (used as AETHER-P3 conditioners)
y_jb   = nan(T, n);  % log10(JB2008 density) at horizons 1..n
y_msis = nan(T, n);  % log10(NRLMSIS density) at horizons 1..n

%% =========================
%  Generate future n-step JB/MSIS estimates
%  For each index i:
%   - The forecast "start" is shifted by history length m:
%       ii = i + m + p,  p = 0..n-1
%   - Baselines are evaluated at the future location/time (ii)
%   - Store log10(density) for use in AETHER-P3 input conditioning
%% =========================
for i = 1 : T - n - m

    % Define the future window indices for this sample
    ii0 = i + m;
    ii1 = i + m + n - 1;

    % Skip sample if any required future value is NaN/Inf
    if any(~isfinite(lat_deg(ii0:ii1))) || any(~isfinite(lon_deg(ii0:ii1))) || any(~isfinite(alt_m(ii0:ii1))) ...
       || any(~isfinite(years(ii0:ii1))) || any(~isfinite(months(ii0:ii1))) || any(~isfinite(days(ii0:ii1))) ...
       || any(~isfinite(hours(ii0:ii1))) || any(~isfinite(minutes(ii0:ii1))) || any(~isfinite(seconds(ii0:ii1))) ...
       || any(~isfinite(lst_h(ii0:ii1)))
        continue;
    end

    jb_vec   = nan(1,n);
    msis_vec = nan(1,n);

    for p = 0 : n-1
        ii = i + m + p;   % future index

        % --- location conversion (deg/m -> rad/km for JB2008 call) ---
        [X,Y,Z] = geodetic2ecef(wgs84, lat_deg(ii), lon_deg(ii), alt_m(ii), 'degrees');
        llh     = ecf2llhT([X;Y;Z]);      % [lat(rad), lon(rad), h(m)]
        lat_rad = llh(1);
        lon_rad = llh(2);
        alt_km  = llh(3)/1000;

        % --- JB2008 baseline at future time & future location ---
        jb_rho = JB2008_at( ...
            years(ii), months(ii), days(ii), hours(ii), minutes(ii), seconds(ii), ...
            lat_rad, lon_rad, alt_km, SOLdata, DTCdata, eopdata);

        % --- NRLMSIS-00 baseline at future time & future location ---
        msis_rho = NRLMSIS00_at( ...
            years(ii), months(ii), days(ii), hours(ii), minutes(ii), seconds(ii), ...
            lat_deg(ii), lon_deg(ii), alt_km, lst_h(ii), swdata, SOLdata, input, flags);

        % Store in log10 domain (consistent with your ML pipeline)
        jb_vec(p+1)   = log10(jb_rho);
        msis_vec(p+1) = log10(msis_rho);
    end

    y_jb(i,:)   = jb_vec;
    y_msis(i,:) = msis_vec;
end

%% =========================
%  Keep only rows with finite values across all horizons.
%% =========================
keep = all(isfinite(y_jb), 2) & all(isfinite(y_msis), 2);
y_jb   = y_jb(keep,:);
y_msis = y_msis(keep,:);

save('Output/jb2008_pred.mat', 'y_jb', '-v7.3');
save('Output/msis_pred.mat',   'y_msis', '-v7.3');

fprintf('Saved JB2008 predictions: Output/jb2008_pred.mat (%d samples)\n', size(y_jb,1));
fprintf('Saved MSIS  predictions: Output/msis_pred.mat   (%d samples)\n', size(y_msis,1));

%% ==========================================================
%  Functions
%  Thin wrappers around JB2008 and NRLMSISE-00:
%   - Input: future timestamp + future location
%   - Output: density rho (linear domain)
%% ==========================================================
function rho = JB2008_at(y, mo, d, h, mi, s, lat_rad, lon_rad, alt_km, SOLdata, DTCdata, eopdata)
global const PC

% day-of-year and MJD
doy = day(datetime([y, mo, d, h, mi, s]), 'dayofyear');
MJD = Mjday(y, mo, d, h, mi, s);

% === read matched solar indices with lags (JB2008) ===
idx = find(y == SOLdata(1,:) & floor(doy) == SOLdata(2,:), 1);
if isempty(idx)
    error('JB2008_at: No SOL match for %04d-DOY%03d', y, floor(doy));
end
if idx-5 < 1
    error('JB2008_at: SOL index too early for required lags (idx=%d).', idx);
end

SOL = SOLdata(:, idx-1);  % 1-day lag
F10  = SOL(4);  F10B = SOL(5);
S10  = SOL(6);  S10B = SOL(7);

SOL = SOLdata(:, idx-2);  % 2-day lag
XM10  = SOL(8);  XM10B = SOL(9);

SOL = SOLdata(:, idx-5);  % 5-day lag
Y10   = SOL(10); Y10B = SOL(11);

% === DTC ===
idxD = (y == DTCdata(1,:) & floor(doy) == DTCdata(2,:));
if ~any(idxD)
    error('JB2008_at: No DTC match for %04d-DOY%03d', y, floor(doy));
end
DTC = DTCdata(:, idxD);
DSTDTC = DTC(floor(h) + 3);

% === EOP / time conversions ===
[x_pole,y_pole,UT1_UTC,LOD,dpsi,deps,dx_pole,dy_pole,TAI_UTC] = IERS_JB(eopdata, MJD, 'l');
[UT1_TAI,UTC_GPS,UT1_GPS,TT_UTC,GPS_UTC] = timediff(UT1_UTC, TAI_UTC);
[DJMJD0, DATE] = iauCal2jd(y, mo, d);
TIME = (60*(60*h + mi) + s)/86400;
UTC  = DATE + TIME;
TT   = UTC + TT_UTC/86400;
TUT  = TIME + UT1_UTC/86400;
UT1  = DATE + TUT;
GWRAS = iauGmst06(DJMJD0, UT1, DJMJD0, TT);

% SAT state (JB expects: RA-lon, lat, alt)
SAT = zeros(1,3);
SAT(1) = mod(GWRAS + lon_rad, 2*pi);
SAT(2) = lat_rad;
SAT(3) = alt_km;

% Sun position
MJD_TDB = Mjday_TDB(TT);
[~,~,~,~,~,~,~,~,~,~,r_Sun,~] = JPL_Eph_DE430(MJD_TDB);
ra_Sun  = atan2(r_Sun(2), r_Sun(1));
dec_Sun = atan2(r_Sun(3), sqrt(r_Sun(1)^2 + r_Sun(2)^2));
SUN = [ra_Sun, dec_Sun];

[~, rho] = JB2008(MJD, SUN, SAT, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC);
end

function rho = NRLMSIS00_at(y, mo, d, h, mi, s, lat_deg, lon_deg, alt_km, lst_h, swdata, SOLdata, input, flags)
% day-of-year
doy = day(datetime([y, mo, d, h, mi, s]), 'dayofyear');

% fill input
input.year   = y;
input.doy    = doy;
input.sec    = h*3600 + mi*60 + s;
input.alt    = alt_km;
input.g_lat  = lat_deg;
input.g_long = lon_deg;
input.lst    = lst_h;

% pick matching swdata row (y,mo,d)
idx_sw = (y == swdata(1,:)) & (mo == swdata(2,:)) & (d == swdata(3,:));
if ~any(idx_sw)
    error('NRLMSIS00_at: No swdata match for %04d-%02d-%02d', y, mo, d);
end
sw = swdata(:, idx_sw);

% SOL (F10.7 / F10.7A)
idx_sol = (y == SOLdata(1,:)) & (doy == SOLdata(2,:));
if ~any(idx_sol)
    error('NRLMSIS00_at: No SOL match for %04d-DOY%03d', y, doy);
end
input.f107  = SOLdata(4, idx_sol);
input.f107A = SOLdata(5, idx_sol);

% ap indices (3-hour slots)
if h < 3
    input.ap_a.a(1) = sw(15);
elseif h < 6
    input.ap_a.a(2) = sw(16);
elseif h < 9
    input.ap_a.a(3) = sw(17);
elseif h < 12
    input.ap_a.a(4) = sw(18);
elseif h < 15
    input.ap_a.a(5) = sw(19);
elseif h < 18
    input.ap_a.a(6) = sw(20);
elseif h < 21
    input.ap_a.a(7) = sw(21);
else
    input.ap_a.a(8) = sw(22);
end
input.ap = sw(23);

% run model
Esti = nrlmsise00(input, flags);

% Depending on your modified NRLMSISE-00 implementation:
% - some versions return a struct with fields (e.g., Esti.d)
% - your pipeline expects a scalar density
rho = Esti;
end
