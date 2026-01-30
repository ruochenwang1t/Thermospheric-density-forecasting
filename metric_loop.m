%% ========================================================================
%  AETHER-P3 Forecast Evaluation (Seed-Averaged)
%
%  Purpose:
%   This script evaluates AETHER-P3 probabilistic thermospheric density forecasts
%   using multiple random seeds, then averages predictions
%   and uncertainties across seeds before computing metrics and plotting.
%
%  Domains:
%   - Model outputs are in log10(density).
%   - Metrics:
%       * Pearson R / R^2, RMSE, Relative Error computed in linear domain (rho).
%       * Coverage Rate computed in log10 domain using mu ± 2*sigma.
%       * MACE computed from two-sided Gaussian prediction intervals in log10 domain.
%
%  Plots:
%   1) Time series (example horizon) with 2σ uncertainty band in linear domain
%   2) Overall scatter density plot (all horizons)
%   3) Scatter density plots per selected horizons
%   4) Calibration curves + MACE
% ========================================================================

clear; close all;

%% --------------------------- Config -------------------------------------
test = 1;                 % Test case index (e.g., 1-8)
resolution = 10;          % Time resolution for plotting (minutes)
output_horizon = 36;      % Forecast horizon length (steps)
N = 6;                    % Number of subsets (optional use for subplots)
gap = output_horizon / N; % kept for compatibility (currently unused)
i_index = [6, 12, 18, 24, 30, 36]; % Horizons (steps) to report and plot
plot_figure = 1;          % 1: generate figures, 0: skip plotting
save_plot = 0;            % 1: export figures to disk, 0: do not save

% -------------------- Multi-seed setting (AETHER-P3) ---------------------
% Each seed corresponds to an independent training run / initialization.
% We average mu and variance across seeds to obtain a seed-robust estimate.
seeds = 10:19;            % Seeds to include in the ensemble average
S = numel(seeds);

%% --------------------------- Load data ----------------------------------
% -------------------- Load AETHER-P3 outputs across seeds ----------------
% For each seed, load:
%   mu       : predictive mean (log10 domain)
%   variance : predictive variance (log10 domain)
%   y_true   : truth (log10 domain) - expected identical across seeds
mu_all  = [];
var_all = [];
y_true_ref = [];

for k = 1:S
    sd = seeds(k);

    baseDir = fullfile('External_Test', sprintf('Seed_%d', sd), sprintf('Test%d', test));
    muPath  = fullfile(baseDir, 'mu.mat');      % log10 domain (after inverse normalization)
    varPath = fullfile(baseDir, 'var.mat');     % variance in log10 domain
    yPath   = fullfile(baseDir, 'y_true.mat');  % truth in log10 domain

    if ~(isfile(muPath) && isfile(varPath) && isfile(yPath))
        error('Missing external-test files: Seed=%d Test=%d under %s', sd, test, baseDir);
    end

    A = load(muPath);      % expects variable: mu
    B = load(varPath);     % expects variable: variance
    C = load(yPath);       % expects variable: y_true

    if ~isfield(A,'mu') || ~isfield(B,'variance') || ~isfield(C,'y_true')
        error('Bad mat fields in %s (need variables: mu / variance / y_true).', baseDir);
    end

    mu_k  = double(A.mu);
    var_k = double(B.variance);
    yt_k  = double(C.y_true);

    % Initialize 3D stacks after reading the first seed
    if isempty(mu_all)
        [T, n] = size(mu_k);
        mu_all  = nan(T, n, S);
        var_all = nan(T, n, S);
        y_true_ref = yt_k;
    else
        % Enforce consistent shapes across seeds
        if ~isequal(size(mu_k), [T, n]) || ~isequal(size(var_k), [T, n])
            error('Shape mismatch at Seed=%d: expected (%d,%d).', sd, T, n);
        end
        % Enforce identical y_true across seeds (to avoid silent mix-ups)
        if ~isequal(size(yt_k), size(y_true_ref))
            error('y_true shape mismatch at Seed=%d.', sd);
        end
        mask_cmp = isfinite(yt_k(:)) & isfinite(y_true_ref(:));
        if any(yt_k(mask_cmp) ~= y_true_ref(mask_cmp))
            error('y_true value mismatch detected at Seed=%d (y_true should be identical across seeds).', sd);
        end
    end

    mu_all(:,:,k)  = mu_k;
    var_all(:,:,k) = var_k;
end

% -------------------- Seed-averaged AETHER-P3 outputs --------------------
mu       = mean(mu_all, 3, 'omitnan');      % log10 mean forecast
variance = mean(var_all, 3, 'omitnan');     % log10 variance (seed-avg)
y_true   = y_true_ref;                     % log10 truth

% Loads kept as-is (used for building time axis based on raw_data)
load(sprintf('Output/Model/Model1/Test/Test%d/raw_data.mat', test))
load(sprintf('Output/Model/Model1/Test/Test%d/ydata.mat', test))

%% ------------------------ Case switch -----------------------------------
% Select the column used to reconstruct the raw observation vector y_ori
% for matching indices to ydata.
switch test
    case {5}
        y_ori = log10(save_data(:, 11));
    case {1, 2, 3, 4, 6, 7, 8}
        y_ori = log10(save_data(:, 7));
end

%% ------------------------ Data process ----------------------------------
[T, n]  = size(mu);

% Per-sample filter in log10 domain:
% - Require finite mu/variance/y_true
% - Clip truth to a reasonable physical range in log10 (adjust as needed)
valid_mask = isfinite(mu) & isfinite(y_true) & isfinite(variance) & ...
    (y_true >= -15) & (y_true <= 0);

mu(~valid_mask)       = NaN;
y_true(~valid_mask)   = NaN;
variance(~valid_mask) = NaN;

% Convert to linear domain for linear-metrics (rho in kg/m^3 or equivalent)
rho_pred = 10.^mu;
rho_true = 10.^y_true;

sigma = sqrt(variance);  % sigma in log10 domain

%% ------------------------ Metrics (linear domain) -----------------------
% Pearson R overall and per-horizon; R^2 defined as r^2
R = corrcoef(rho_true(~isnan(rho_true)), rho_pred(~isnan(rho_true)));
R = R(1,2);
R2_overall = R.^2;

R_list = zeros(1,n);
R2_list = zeros(1,n);
for i = 1:n
    x = rho_true(:, i);
    y = rho_pred(:, i);
    RR = corrcoef(x(~isnan(x)), y(~isnan(y)));
    R_list(i)  = RR(1,2);
    R2_list(i) = R_list(i).^2;
end

%% RMSE (overall: average across time, then average across samples)
err = rho_true - rho_pred;
RMSE = mean(sqrt(mean(err.^2, 2, 'omitnan')), 'omitnan');
RMSE_list = sqrt(mean(err.^2, 1, 'omitnan'));  % per-horizon RMSE

%% Coverage rate (log10 domain, 2-sigma)
% Checks if truth lies within mu ± 2*sigma at each (t,h) point.
inside = abs(y_true - mu) <= 2.*sigma & valid_mask;     % (T x n)
CovRate = sum(inside(:)) ./ sum(valid_mask(:));
CovRate_list = sum(inside,1) ./ sum(valid_mask, 1);

%% MACE (log10 domain; two-sided Gaussian intervals)
% Computes empirical coverage EC(CL) for multiple nominal coverages CL.
CL = [0.05:0.05:0.95 0.96 0.99];
z  = sqrt(2) * erfinv(CL);
K  = numel(CL);

mu3d = mu     + zeros(1,1,K);
sg3d = sigma  + zeros(1,1,K);
y3d  = y_true + zeros(1,1,K);
z3d  = reshape(z, 1, 1, []);

lb = mu3d - z3d .* sg3d;
ub = mu3d + z3d .* sg3d;

valid3 = isfinite(y3d) & isfinite(mu3d) & isfinite(sg3d);
in3    = (y3d >= lb) & (y3d <= ub) & valid3;

num_in = squeeze(sum(in3, 1));    % n x K
den_in = squeeze(sum(valid3, 1)); % n x K
den_in(den_in == 0) = NaN;

EC_t      = num_in ./ den_in;                       % n x K (per-horizon EC)
MACE_list = mean(abs(EC_t - CL), 2, 'omitnan');     % n x 1

EC_den = sum(den_in, 1); EC_den(EC_den == 0) = NaN;
EC   = sum(num_in, 1) ./ EC_den;                   % 1 x K (overall EC)
MACE = mean(abs(EC - CL), 'omitnan');

%% Relative Error (linear domain)
relativeError = abs(rho_true - rho_pred) ./ rho_true;
RE_list = mean(relativeError, 1, 'omitnan');   % per-horizon
RE = mean(RE_list, 'omitnan');                 % overall

%% ------------------------ Print metrics ---------------------------------
fprintf('The Overall AETHER-P3 Performance (Seed-Avg) of Test %d:\n', test);
fprintf('Seeds: %s\n', mat2str(seeds));
fprintf('R: %.4f   R^2: %.4f\n', R, R2_overall);
fprintf('RMSE: %.4e\n', RMSE);
fprintf('Coverage Rate: %.4f\n', CovRate);
fprintf('MACE: %.4f\n', MACE);
fprintf('Relative Error: %.4f\n\n', RE);

for i = i_index
    fprintf('Overall: t+%d -> R: %.4f  R^2: %.4f  RMSE: %.4e  CovRate: %.4f  MACE: %.4f  RE: %.4f\n', ...
        i, R_list(i), R2_list(i), RMSE_list(i), CovRate_list(i), MACE_list(i), RE_list(i));
end
fprintf('\n');

%% ------------------------ Build time axis / Plotting --------------------
if plot_figure == 1
    % Build a datetime axis by matching y_ori values to ydata(:,1) (existing workflow)
    index = find(ismember(y_ori, ydata(:, 1)));
    date = datetime(save_data(index, 1:6));
    date_seq = date + minutes((0:n-1)*resolution);

    % Convert log10 uncertainty band into linear domain band for plotting
    rho_lb = 10.^(mu - 2*sigma);
    rho_ub = 10.^(mu + 2*sigma);

    %% ------------------------ Figure 1: Time series (example horizon) ---
    % NOTE: Currently this plots only horizon i=36.
    figure(1); clf
    for i = 36
        hold on; grid on
        valid_index = find(valid_mask(:, i));

        plot(date_seq(valid_index, i), rho_true(valid_index, i), 'b-', 'LineWidth',1);
        plot(date_seq(valid_index, i), rho_pred(valid_index, i), 'r-', 'LineWidth',1);
        fill([date_seq(valid_index, i); flipud(date_seq(valid_index, i))], ...
            [rho_lb(valid_index, i);  flipud(rho_ub(valid_index, i))], ...
            [.8 0.2 0.2], 'EdgeColor','none', 'FaceAlpha',0.3);

        xlabel('Date'); ylabel('Density');
    end
    legend({'Observations','AETHER-P3 Forecast', '\mu \pm 2\sigma'}, FontSize=20);
    title(sprintf('Test %d: AETHER-P3 Forecast vs Observations', test), FontSize=20);
    set(gca,'FontSize',20);

    %% ------------------------ Figure 2: Overall scatter (all horizons) ---
    figure(2); clf; hold on; grid on;

    x_all = rho_true(:);
    y_all = rho_pred(:);
    v_all = isfinite(x_all) & isfinite(y_all);
    x_all = x_all(v_all);
    y_all = y_all(v_all);

    lo = min(min(x_all), min(y_all));
    hi = max(max(x_all), max(y_all));

    nbin = 300;
    edgesX = linspace(lo, hi, nbin+1);
    edgesY = linspace(lo, hi, nbin+1);
    edgesX(end) = edgesX(end) + eps(edgesX(end));
    edgesY(end) = edgesY(end) + eps(edgesY(end));

    [N2,~,~,binX,binY] = histcounts2(x_all, y_all, edgesX, edgesY);
    good = (binX >= 1 & binX <= nbin) & (binY >= 1 & binY <= nbin);
    xg = x_all(good); yg = y_all(good);
    binX = binX(good); binY = binY(good);
    idx = sub2ind([nbin nbin], binX, binY);
    c   = log10(N2(idx) + 1);

    scatter(xg, yg, 10, c, 'filled');
    colormap(jet);
    cb = colorbar('Location','eastoutside');
    cb.Label.FontSize = 20;

    plot([lo hi],[lo hi],'k-','LineWidth',2);

    % OLS fit (with intercept) for visual reference
    p_all = polyfit(xg, yg, 1);
    plot([lo hi], polyval(p_all,[lo hi]), '--r','LineWidth',2);

    % R^2 as Pearson r^2
    r_all  = corr(xg, yg);
    R2_all = r_all.^2;

    axis square; xlim([lo hi]); ylim([lo hi]);
    set(gca,'FontSize',18);
    xlabel('Observation'); ylabel('Prediction');
    title(sprintf('Test %d: Overall scatter (all horizons)', test));
    text(0.02, 0.98, sprintf('R^2=%.4f', R2_all), ...
        'Units','normalized','VerticalAlignment','top','FontSize',20,'BackgroundColor','w');

    %% ------------------------ Figure 3: Scatter per selected horizons ----
    figure(3); clf

    % pooled finite points for global edges
    x_all = rho_true(:);
    y_all = rho_pred(:);
    v_all = isfinite(x_all) & isfinite(y_all);
    x_all = x_all(v_all);
    y_all = y_all(v_all);

    if isempty(x_all)
        warning('No finite points to plot for per-horizon scatter.');
    else
        lo = min(min(x_all), min(y_all));
        hi = max(max(x_all), max(y_all));

        nbin = 200;
        edgesX = linspace(lo, hi, nbin+1);
        edgesY = linspace(lo, hi, nbin+1);
        edgesX(end) = edgesX(end) + eps(edgesX(end));
        edgesY(end) = edgesY(end) + eps(edgesY(end));

        panel_idx = i_index;
        P = numel(panel_idx);
        ncol = min(3, P);
        nrow = ceil(P / ncol);
        tiledlayout(nrow, ncol, 'Padding','compact','TileSpacing','compact');

        % Unified color scale across panels (max bin count)
        maxCount = 1;
        for kk = 1:P
            i = panel_idx(kk);
            xi = rho_true(:, i); yi = rho_pred(:, i);
            v  = isfinite(xi) & isfinite(yi);
            xi = xi(v); yi = yi(v);
            if isempty(xi), continue; end
            Ntmp = histcounts2(xi, yi, edgesX, edgesY);
            mc = max(Ntmp(:));
            if isfinite(mc)
                maxCount = max(maxCount, mc);
            end
        end
        clim_val = [0, log10(maxCount + 1)];

        for kk = 1:P
            i = panel_idx(kk);
            nexttile; hold on; grid on;

            xi = rho_true(:, i); yi = rho_pred(:, i);
            v  = isfinite(xi) & isfinite(yi);
            xi = xi(v); yi = yi(v);
            if isempty(xi)
                title(sprintf('t+%d (no valid data)', i));
                axis square; box on;
                continue;
            end

            [N_i,~,~,binX,binY] = histcounts2(xi, yi, edgesX, edgesY);
            good = (binX >= 1 & binX <= nbin) & (binY >= 1 & binY <= nbin);
            xi2 = xi(good); yi2 = yi(good);
            binX = binX(good); binY = binY(good);
            idx = sub2ind([nbin nbin], binX, binY);
            c   = log10(N_i(idx) + 1);

            scatter(xi2, yi2, 10, c, 'filled');

            plot([lo hi], [lo hi], 'k-', 'LineWidth', 2);  % 1:1 line

            if numel(xi2) >= 2
                p = polyfit(xi2, yi2, 1);
                plot([lo hi], polyval(p,[lo hi]), '--r', 'LineWidth', 2);
                r_h  = corr(xi2, yi2);
                R2_h = r_h.^2;
                text(0.05, 0.95, sprintf('$R^2 = %.4f$', R2_h), ...
                    'Units','normalized','Interpreter','latex','FontSize',20);
            end

            axis square; xlim([lo hi]); ylim([lo hi]);
            colormap(jet); caxis(clim_val);
            title(sprintf('t+%d', i));
            xlabel('Observation'); ylabel('Prediction');
            set(gca,'FontSize',20);
        end

        cb = colorbar('Location','eastoutside');
        cb.Label.FontSize = 20;
        sgtitle(sprintf('Test %d: Scatter density per horizon (True vs AETHER-P3)', test),'FontSize',20);
    end

    %% ------------------------ Figure 4: Calibration curves ----------------
    figure(4); clf; hold on; grid on;

    % EC_t is n x K (per-horizon), EC_mean is averaged across horizons
    EC_mean   = mean(EC_t, 1);
    MACE_mean = mean(abs(EC_mean - CL));

    for j = i_index
        plot(CL, EC_t(j,:), '-', 'LineWidth', 0.8);
    end
    plot(CL, EC_mean, 'o-', 'LineWidth', 2);
    plot([0 1],[0 1],'k--','LineWidth',2);
    axis([0 1 0 1]); axis square;
    xlabel('Nominal coverage'); ylabel('Empirical coverage');
    title(sprintf('Test %d: Calibration Curves Across the Forecast Horizon', test));
    labels = [ arrayfun(@(kk) sprintf('t+%d', kk), i_index, 'UniformOutput', false), ...
        {'Mean','Ideal'} ];
    legend(labels, 'Location', 'southeast');
    set(gca,'FontSize',18);
    text(0.02, 0.96, sprintf('MACE = %.4f', MACE), ...
        'Units','normalized','FontSize',20,'VerticalAlignment','top','BackgroundColor','w');

    %% ------------------------ Save figures --------------------------------
    if save_plot == 1
        outdir = fullfile('External_Test', sprintf('SeedAvg_Seed%d_%d', seeds(1), seeds(end)), sprintf('Test%d', test));
        if ~exist(outdir,'dir'); mkdir(outdir); end

        figNums = [1 2 3 4];
        names   = {'TruthPred_all','Scatter_overall','Scatter_by_horizon','Calibration'};

        for kk = 1:numel(figNums)
            fn = figNums(kk);
            if ishghandle(fn)
                f = figure(fn);
                base = fullfile(outdir, sprintf('Test%d_%s', test, names{kk}));
                exportgraphics(f, [base '.png'], 'Resolution', 300, 'BackgroundColor','white');
            end
        end
    end
end
