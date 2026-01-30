%% ============================================================
%  Compare WAM-IPE vs EDL (seed-averaged) on a given Test case
%  - Loads WAM-IPE nearest & interpolation predictions (physical domain)
%  - Loads EDL mu (log10 domain) for seeds 10..19, averages in log10,
%    converts to physical density for skill metrics
%  - Aligns series by datetime robustly using timetables
%% ============================================================

clear; close all; clc;

%% ------------------------ User settings ------------------------
test  = 8;              % Test index
h_idx = 1;              % horizon step to evaluate (1-based)
n = 18;                 % input horizon
m = 36;                 % output horizon
seeds = 10:19;
S = numel(seeds);

% Paths
testDir = fullfile("Test", sprintf("Test%d", test));

% WAM-IPE CSVs
wamNearCSV   = fullfile(testDir, "WAM_IPE_Prediction_Nearest.csv");
wamInterpCSV = fullfile(testDir, "WAM_IPE_Prediction_Interpolation.csv");

% Raw data MAT (contains save_data with datetime columns and rho_true column)
rawMat = fullfile(testDir, "raw_data.mat");

% External EDL outputs base
extBase = "External_Test";

%% ------------------------ Load WAM-IPE ------------------------
raw_data_nearest = readtable(wamNearCSV);
rho_pred_near    = raw_data_nearest.pred_density;

raw_data_interp  = readtable(wamInterpCSV);
rho_pred_interp  = raw_data_interp.pred_density;

%% ------------------------ Load truth + timestamps ------------------------
load(rawMat, "save_data");

% NOTE: choose the correct column for rho_true from save_data
% rho_true = save_data(:, 11); % GRACE-FO (example)
rho_true = save_data(:, 7);    % SWARM (your current choice)

date_all = datetime(save_data(:,1:6));   % [Y M D h m s] -> datetime
date_wamipe = date_all;                 % initial timeline for WAM-IPE arrays

%% ------------------------ Load EDL outputs for all seeds ------------------------
mu_stack  = [];
ytrue_ref = [];

for k = 1:S
    sd = seeds(k);

    baseDir = fullfile(extBase, sprintf("Seed_%d", sd), sprintf("Test%d", test));

    mu_file = fullfile(baseDir, "mu.mat");        % expects variable: mu (log10)
    yt_file = fullfile(baseDir, "y_true.mat");    % expects variable: y_true (log10)

    if ~(isfile(mu_file) && isfile(yt_file))
        error("Missing EDL files for Seed=%d Test=%d under %s", sd, test, baseDir);
    end

    A = load(mu_file);
    B = load(yt_file);

    if ~isfield(A, "mu") || ~isfield(B, "y_true")
        error("Bad mat fields under %s (need mu and y_true).", baseDir);
    end

    mu_k = double(A.mu);
    yt_k = double(B.y_true);

    if isempty(mu_stack)
        [T, H] = size(mu_k); %#ok<ASGLU>
        mu_stack = nan(T, H, S);
        mu_stack(:,:,k) = mu_k;
        ytrue_ref = yt_k;
    else
        if ~isequal(size(mu_k), size(mu_stack(:,:,1)))
            error("mu shape mismatch at Seed=%d (Test=%d).", sd, test);
        end
        if ~isequal(size(yt_k), size(ytrue_ref))
            error("y_true shape mismatch at Seed=%d (Test=%d).", sd, test);
        end

        % Enforce y_true identical across seeds (after removing NaNs)
        mask_cmp = isfinite(yt_k(:)) & isfinite(ytrue_ref(:));
        if any(yt_k(mask_cmp) ~= ytrue_ref(mask_cmp))
            error("y_true mismatch detected at Seed=%d (Test=%d).", sd, test);
        end

        mu_stack(:,:,k) = mu_k;
    end
end

% Average mu in log10 domain (recommended for log outputs)
mu_avg_log10 = mean(mu_stack, 3, "omitnan");

% Select a horizon column
mu_pred_log10   = mu_avg_log10(:, h_idx);
y_true_log10    = ytrue_ref(:, h_idx);

% Convert EDL to physical domain for R/RMSE/RE
rho_pred_EDL = 10.^mu_pred_log10;
rho_true_EDL = 10.^y_true_log10;

%% ------------------------ Define EDL timestamps ------------------------
% IMPORTANT:
% That implies your EDL samples correspond to a
% subset of date_all (windowing offset). Keep it, but isolate it clearly here.
date_edl = date_all(n+1:end-m);

% Sanity check lengths: date_edl must match length of mu_pred_log10
if numel(date_edl) ~= numel(mu_pred_log10)
    warning("Length mismatch: date_edl (%d) vs EDL series (%d). Check your offset 19:end-36.", ...
        numel(date_edl), numel(mu_pred_log10));
end

%% ------------------------ Filter WAM-IPE ------------------------
% - Remove where predictions are 0, >1, or NaN
% - Remove where truth > 1
nan_index_near   = find(rho_pred_near==0 | rho_pred_near>1 | isnan(rho_pred_near));
nan_index_interp = find(rho_pred_interp==0 | rho_pred_interp>1 | isnan(rho_pred_interp));
nan_index_true   = find(rho_true>1);

index = unique([nan_index_near; nan_index_interp; nan_index_true]);

rho_true(index)        = [];
rho_pred_near(index)   = [];
rho_pred_interp(index) = [];
date_wamipe(index)     = [];

%% ------------------------ Filter EDL ------------------------
% Filter in physical domain (consistent with your comparison constraints)
valid_mask_edl = isfinite(rho_true_EDL) & rho_true_EDL > 0 & rho_true_EDL < 1 & ...
                 isfinite(rho_pred_EDL) & rho_pred_EDL > 0 & rho_pred_EDL < 1;

rho_true_EDL = rho_true_EDL(valid_mask_edl);
rho_pred_EDL = rho_pred_EDL(valid_mask_edl);

% Keep matching dates
if numel(date_edl) >= numel(valid_mask_edl)
    date_edl = date_edl(valid_mask_edl);
else
    % If mismatch, truncate conservatively
    L = min(numel(date_edl), numel(valid_mask_edl));
    date_edl = date_edl(1:L);
    rho_true_EDL = rho_true_EDL(1:L);
    rho_pred_EDL = rho_pred_EDL(1:L);
end

%% ------------------------ Align by datetime robustly ------------------------
% Build timetables so we can synchronize safely
TT_wam = timetable(date_wamipe, rho_true, rho_pred_near, rho_pred_interp, ...
    "VariableNames", ["rho_true_wam","rho_near","rho_interp"]);

TT_edl = timetable(date_edl, rho_true_EDL, rho_pred_EDL, ...
    "VariableNames", ["rho_true_edl","rho_edl"]);

% Keep only intersection of times
TT = synchronize(TT_wam, TT_edl, "intersection");

% Extract aligned arrays (same timestamps now)
t_common      = TT.date_wamipe;

rho_true_wam  = TT.rho_true_wam;
rho_near      = TT.rho_near;
rho_interp    = TT.rho_interp;

rho_true_edlA = TT.rho_true_edl;
rho_edlA      = TT.rho_edl;

%% ------------------------ Metrics ------------------------
% R (Pearson correlation)
R_near   = corr(rho_true_wam, rho_near,   "Rows","complete");
R_interp = corr(rho_true_wam, rho_interp, "Rows","complete");
R_edl    = corr(rho_true_edlA, rho_edlA,  "Rows","complete");

% RMSE
err_near   = rho_true_wam  - rho_near;
err_interp = rho_true_wam  - rho_interp;
err_edl    = rho_true_edlA - rho_edlA;

RMSE_near   = sqrt(mean(err_near.^2,   "omitnan"));
RMSE_interp = sqrt(mean(err_interp.^2, "omitnan"));
RMSE_edl    = sqrt(mean(err_edl.^2,    "omitnan"));

% Relative Error (mean absolute relative error)
RE_near   = mean(abs(err_near)  ./ rho_true_wam,  "omitnan");
RE_interp = mean(abs(err_interp)./ rho_true_wam,  "omitnan");
RE_edl    = mean(abs(err_edl)   ./ rho_true_edlA, "omitnan");

%% ------------------------ Print results ------------------------
fprintf("Test %d (EDL Seed-Avg %d-%d, horizon=%d):\n", test, seeds(1), seeds(end), h_idx);
fprintf("R:     Near: %.4f  Interp: %.4f  EDL: %.4f\n", R_near, R_interp, R_edl);
fprintf("RMSE:  Near: %.4e  Interp: %.4e  EDL: %.4e\n", RMSE_near, RMSE_interp, RMSE_edl);
fprintf("RE:    Near: %.4f  Interp: %.4f  EDL: %.4f\n\n", RE_near, RE_interp, RE_edl);

%% ------------------------ Plots ------------------------
% (1) Error time series
figure(1); clf;
plot(t_common, err_near,   "-r", "LineWidth", 2); hold on; grid on;
plot(t_common, err_interp, "-b", "LineWidth", 2);
plot(t_common, err_edl,    "--k", "LineWidth", 2);
xlabel("Date");
ylabel("Error ( \rho_{true} - \rho_{pred} )");
title(sprintf("Test %d: Density Error Comparison", test));
legend("WAM-IPE Nearest", "WAM-IPE Interp", "EDL (Seed-Avg)", "Location", "best");
set(gca, "FontSize", 18);

% (2) Density comparison
figure(2); clf;
plot(t_common, rho_true_edlA, "-k", "LineWidth", 2); hold on; grid on;
plot(t_common, rho_edlA,      "-r", "LineWidth", 2);
plot(t_common, rho_interp,    "-",  "LineWidth", 1.5, "Color", [0 0.5 0 0.3]);
xlabel("Date");
ylabel("Density");
title(sprintf("Test %d: Density Comparison", test));
legend("True Density (EDL truth)", "EDL Forecast (Seed-Avg)", "WAM-IPE Interp", "Location", "best");
set(gca, "FontSize", 18);
