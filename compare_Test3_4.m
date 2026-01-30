%% ========================================================================
%  Compare Empirical Models (NRLMSIS-00, JB2008) vs AETHER-P3 (Seed Avg)
%
%  Purpose:
%   - For each selected test case, load:
%       (1) MSIS predictions
%       (2) JB2008 predictions
%       (3) Truth
%       (4) AETHER-P3 predictions from multiple random seeds
%   - Compute overall metrics:
%       * MAE, RMSE, RE
%       * Improvements (%) of AETHER-P3 vs Empirical models
%   - Compute per-horizon metrics (each forecast step)
%
%  Notes:
%   - A single validity mask is applied to all models to ensure fair comparison.
% ========================================================================

clear; close all; clc

%% --------------------------- Paths & Global Settings --------------------
rootModel = "Output/Model/Model1";   % Root folder for MSIS/JB/Truth test outputs
saveDir   = "Parameter";            % Folder to save comparison outputs
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

eps_den  = 1e-20;                   % Small epsilon to avoid division-by-zero in % improvements
m = 18; % input horizon length
n = 36; % output horizon length
% Multi-seed setting for the AETHER-P3 model outputs
seeds = 10:19;
Snum  = numel(seeds);

%% ------------------------------ Loop Through Tests ----------------------
for TestID = 3:4
    fprintf('\n================ Test %d ================\n', TestID);
    try
        %% ----------------------------- Load Inputs ----------------------
        % Baseline predictions and truth (stored under rootModel/Test/TestID/)
        msis_file  = sprintf("%s/Test/Test%d/msis_pred.mat",   rootModel, TestID);
        jb_file    = sprintf("%s/Test/Test%d/jb2008_pred.mat", rootModel, TestID);
        ytrue_file = sprintf("%s/Test/Test%d/ydata.mat",       rootModel, TestID);

        % Date information (kept for time-series plotting)
        date_file  = sprintf("%s/Test/Test%d/raw_data.mat", rootModel, TestID);

        % --- MSIS prediction ---
        S = load(msis_file);
        MSIS_pred = S.y_msis;

        % --- JB2008 prediction ---
        S = load(jb_file);
        JB_pred = S.y_jb;

        % --- Ground truth ---
        S = load(ytrue_file);
        y_true = S.ydata;

        % --- Date vector for plotting ---
        S = load(date_file);
        date = datetime(S.save_data(m+n+1:end, 1:6));

        %% ----------- Load AETHER mu from Multiple Seeds and Average ----------
        mu_all = [];

        for k = 1:Snum
            sd = seeds(k);
            AETHER_file = sprintf("External_Test/Seed_%d/Test%d/mu.mat", sd, TestID);

            if ~isfile(AETHER_file)
                error('Missing AETHER file: %s', AETHER_file);
            end

            S = load(AETHER_file);
            if ~isfield(S,'mu')
                error('AETHER mu file missing variable "mu": %s', AETHER_file);
            end

            mu_k = double(S.mu);

            % Initialize stack on first seed, then validate shape consistency
            if isempty(mu_all)
                mu_all = mu_k;
                [Tmu, Hmu] = size(mu_k);
                mu_stack = nan(Tmu, Hmu, Snum);
                mu_stack(:,:,k) = mu_k;
            else
                if ~isequal(size(mu_k), size(mu_stack(:,:,1)))
                    error('AETHER mu shape mismatch at Seed=%d for Test=%d.', sd, TestID);
                end
                mu_stack(:,:,k) = mu_k;
            end
        end

        % Seed-average AETHER-P3 model prediction
        mu_avg = mean(mu_stack, 3, 'omitnan');

        %% --------------------------- Align Lengths ------------------------
        % Ensure all arrays have identical length N for fair comparisons
        N = min([size(y_true,1), size(MSIS_pred,1), size(JB_pred,1), size(mu_avg,1), numel(date)]);
        y_true    = y_true(1:N, :);
        MSIS_pred = MSIS_pred(1:N, :);
        JB_pred   = JB_pred(1:N, :);
        y_pred    = mu_avg(1:N, :);
        date      = date(1:N);

        %% ----------------------- Valid Mask (Shared) ----------------------
        % Apply ONE mask to all models so computed metrics are directly comparable
        valid_mask = isfinite(y_true) & isfinite(y_pred) & isfinite(MSIS_pred) & isfinite(JB_pred) & ...
                     (y_true >= -20) & (y_true <= 0);

        y_true(~valid_mask)    = NaN;
        y_pred(~valid_mask)    = NaN;
        MSIS_pred(~valid_mask) = NaN;
        JB_pred(~valid_mask)   = NaN;

        %% ---------------------- Errors ---------------------
        % Error defined as truth - prediction
        MSIS_err = y_true - MSIS_pred;
        JB_err   = y_true - JB_pred;
        AETHER_err  = y_true - y_pred;

        %% ---------------- Overall MAE / RMSE ----------------------
        MSIS_MAE  = mean(abs(MSIS_err(:)), 'omitnan');
        JB_MAE    = mean(abs(JB_err(:)),   'omitnan');
        AETHER_MAE   = mean(abs(AETHER_err(:)),  'omitnan');

        MSIS_RMSE = sqrt(mean(MSIS_err(:).^2, 'omitnan'));
        JB_RMSE   = sqrt(mean(JB_err(:).^2,   'omitnan'));
        AETHER_RMSE  = sqrt(mean(AETHER_err(:).^2,  'omitnan'));

        %% ---------------- Relative Error ------------------
        RE_MSIS = mean(abs(1 - 10.^(MSIS_pred(:) - y_true(:))), 'omitnan');
        RE_JB   = mean(abs(1 - 10.^(JB_pred(:)   - y_true(:))), 'omitnan');
        RE_AETHER  = mean(abs(1 - 10.^(y_pred(:)    - y_true(:))), 'omitnan');

        %% ------------- Improvements ------------------
        impr_MAE_vs_MSIS  = 100 * (MSIS_MAE  - AETHER_MAE ) / max(MSIS_MAE,  eps_den);
        impr_RMSE_vs_MSIS = 100 * (MSIS_RMSE - AETHER_RMSE) / max(MSIS_RMSE, eps_den);
        impr_RE_vs_MSIS   = 100 * (RE_MSIS   - RE_AETHER  ) / max(RE_MSIS,   eps_den);

        impr_MAE_vs_JB    = 100 * (JB_MAE    - AETHER_MAE ) / max(JB_MAE,    eps_den);
        impr_RMSE_vs_JB   = 100 * (JB_RMSE   - AETHER_RMSE) / max(JB_RMSE,   eps_den);
        impr_RE_vs_JB     = 100 * (RE_JB     - RE_AETHER  ) / max(RE_JB,     eps_den);

        %% ----------- Per-horizon Metrics & Improvements (Optional) --------
        % Compute metrics independently for each horizon/step k
        nH = size(y_true, 2);

        for k = 1:nH
            e_msis = y_true(:,k) - MSIS_pred(:,k);
            e_jb   = y_true(:,k) - JB_pred(:,k);
            e_AETHER  = y_true(:,k) - y_pred(:,k);

            % MAE/RMSE
            mae_msis  = mean(abs(e_msis), 'omitnan');
            rmse_msis = sqrt(mean(e_msis.^2, 'omitnan'));
            mae_jb    = mean(abs(e_jb),   'omitnan');
            rmse_jb   = sqrt(mean(e_jb.^2, 'omitnan'));
            mae_AETHER   = mean(abs(e_AETHER),  'omitnan');
            rmse_AETHER  = sqrt(mean(e_AETHER.^2, 'omitnan'));

            % RE
            re_msis   = mean(abs(1 - 10.^(MSIS_pred(:,k) - y_true(:,k))), 'omitnan');
            re_jb     = mean(abs(1 - 10.^(JB_pred(:,k)   - y_true(:,k))), 'omitnan');
            re_AETHER    = mean(abs(1 - 10.^(y_pred(:,k)    - y_true(:,k))), 'omitnan');

            % Improvements (%)
            impr_mae_msis  = 100 * (mae_msis  - mae_AETHER) / max(mae_msis,  eps_den);
            impr_rmse_msis = 100 * (rmse_msis - rmse_AETHER) / max(rmse_msis, eps_den);
            impr_re_msis   = 100 * (re_msis   - re_AETHER ) / max(re_msis,   eps_den);

            impr_mae_jb    = 100 * (mae_jb    - mae_AETHER) / max(mae_jb,    eps_den);
            impr_rmse_jb   = 100 * (rmse_jb   - rmse_AETHER) / max(rmse_jb,   eps_den);
            impr_re_jb     = 100 * (re_jb     - re_AETHER ) / max(re_jb,     eps_den);
        end

        %% --------------------------- Console Summary ----------------------
        fprintf('Overall (log10 domain):\n');
        fprintf('MSIS  MAE=%.3e  RMSE=%.3e\n', MSIS_MAE, MSIS_RMSE);
        fprintf('JB    MAE=%.3e  RMSE=%.3e\n', JB_MAE,   JB_RMSE);
        fprintf('AETHER (Seed-Avg %d-%d)  MAE=%.3e  RMSE=%.3e\n', seeds(1), seeds(end), AETHER_MAE, AETHER_RMSE);

        fprintf('Linear RE:  MSIS=%.4f  JB=%.4f  AETHER=%.4f\n', RE_MSIS, RE_JB, RE_AETHER);

        fprintf('Improvements (AETHER vs MSIS): MAE=%.2f%%  RMSE=%.2f%%  RE=%.2f%%\n', ...
            impr_MAE_vs_MSIS, impr_RMSE_vs_MSIS, impr_RE_vs_MSIS);
        fprintf('Improvements (AETHER vs JB):   MAE=%.2f%%  RMSE=%.2f%%  RE=%.2f%%\n', ...
            impr_MAE_vs_JB, impr_RMSE_vs_JB, impr_RE_vs_JB);

        %% ---------------- Plot: RMSE over Time (Horizon-Avg) --------------
        % Horizon-averaged RMSE at each time index
        rmse_t_msis = sqrt(mean(MSIS_err.^2, 2, 'omitnan'));
        rmse_t_jb   = sqrt(mean(JB_err.^2,   2, 'omitnan'));
        rmse_t_AETHER  = sqrt(mean(AETHER_err.^2,  2, 'omitnan'));

        figure('Name', sprintf('Test%d RMSE over horizons (log10)', TestID), 'Color', 'w');
        plot(date, rmse_t_msis, 'r-'); hold on;
        plot(date, rmse_t_jb,   'k-');
        plot(date, rmse_t_AETHER,  'b-');
        legend('NRLMSIS-00','JB2008','Global Forecasting Model');
        xlabel('Date');
        ylabel('RMSE over horizons (log10 domain)');
        grid on;
        title(sprintf('Test %d: Model Accuracy Comparison', TestID));
        ax = gca; ax.FontSize = 20;

        saveas(gcf, sprintf('%s/compare_Test%d_MSIS_JB_AETHER_SeedAvg_%d_%d.png', ...
            saveDir, TestID, seeds(1), seeds(end)));

    catch ME
        % Continue to next TestID if any required file is missing or malformed
        fprintf('Test%d failed: %s\n', TestID, ME.message);
    end
end
