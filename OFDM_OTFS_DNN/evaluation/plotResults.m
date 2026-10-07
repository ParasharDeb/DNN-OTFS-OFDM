function plotResults(cfg, ds, data, model, results, summary, timing)
%PLOTRESULTS All project figures (PNG + FIG) in cfg.paths.plots.
%
%   01  CSI / channel representation (TF magnitude, ICI matrix)
%   02  Distribution of the optimal OFDM allocation (+ vs SNR and Doppler)
%   03  Training and validation loss
%   04  Validation performance during training
%   05  Predicted vs actual N_OFDM (row-normalised confusion matrix)
%   06  Allocation error histogram
%   07  BER vs SNR             (optimal, DNN, fixed baselines)
%   08  Spectral efficiency vs SNR
%   09  DNN vs exhaustive search (score per SNR, regret CDF, % of optimum)
%   10  Computation time comparison
%   11  Performance vs computation-time trade-off
%   12  Detection probability vs target echo SNR
%   13  Mean allocation vs Doppler: optimal vs DNN

if ~exist(cfg.paths.plots, 'dir'), mkdir(cfg.paths.plots); end
cand = ds.candidates;
K    = cfg.sys.N_total;
sty  = ruleStyles(results.bestFixedNofdm);

%% 01 ---------------------------------------------------------- CSI representation
mob = ds.meta.fDmax .* ds.meta.dynFraction;
[~, iLow]  = min(mob + 1e6 * (ds.meta.snr_dB < 20));          % static, high SNR
[~, iHigh] = max(mob - 1e6 * (ds.meta.snr_dB < 20));          % strongly dynamic, high SNR
sL = regenerateSample(cfg, ds.meta.sampleId(iLow));
sH = regenerateSample(cfg, ds.meta.sampleId(iHigh));
fig = newFig('CSI representation', [60 60 1150 820]);
tl  = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl);
imagesc(ax, 1:cfg.sys.Nsym, 1:K, 20 * log10(abs(sL.obs.Hest)));  axis(ax, 'xy'); cb = colorbar(ax); cb.Label.String = '|H| (dB)';
styleAxes(ax, 'OFDM symbol index', 'Subcarrier index', sprintf('Estimated CSI |H(k,n)|: static (f_D=%.0f Hz, %.0f%% moving)', sL.ch.fDmax, 100 * sL.ch.dynFraction));
ax = nexttile(tl);
imagesc(ax, 1:cfg.sys.Nsym, 1:K, 20 * log10(abs(sH.obs.Hest)));  axis(ax, 'xy'); cb = colorbar(ax); cb.Label.String = '|H| (dB)';
styleAxes(ax, 'OFDM symbol index', 'Subcarrier index', sprintf('Estimated CSI |H(k,n)|: dynamic (f_D=%.0f Hz, %.0f%% moving)', sH.ch.fDmax, 100 * sH.ch.dynFraction));
ax = nexttile(tl);
Hm = 20 * log10(abs(sH.H(:, :, 1)) + 1e-12);
imagesc(ax, 1:K, 1:K, Hm, max(Hm(:)) + [-50 0]);  axis(ax, 'ij'); cb = colorbar(ax); cb.Label.String = '|H(k,j)| (dB)';
styleAxes(ax, 'Transmit subcarrier j', 'Receive subcarrier k', 'TF channel matrix of symbol 1 (dynamic case): off-diagonal = ICI');
ax = nexttile(tl); hold(ax, 'on');
plot(ax, 1:K, 20 * log10(abs(diag(sL.H(:, :, 1)))), 'LineWidth', 1.5);
plot(ax, 1:K, 20 * log10(abs(diag(sH.H(:, :, 1)))), 'LineWidth', 1.5);
iciL = 10 * log10(mean(sum(abs(sL.H(:, :, 1)).^2, 2) - abs(diag(sL.H(:, :, 1))).^2) + 1e-12);
iciH = 10 * log10(mean(sum(abs(sH.H(:, :, 1)).^2, 2) - abs(diag(sH.H(:, :, 1))).^2) + 1e-12);
yline(ax, iciH, '--', sprintf('ICI power, dynamic: %.1f dB', iciH), 'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');
styleAxes(ax, 'Subcarrier index', 'Channel gain (dB)', 'Frequency response, symbol 1 (true CSI)');
legend(ax, {sprintf('static (ICI %.0f dB)', iciL), sprintf('dynamic (ICI %.0f dB)', iciH)}, 'Location', 'southwest');
saveFigure(fig, cfg, '01_csi_representation');

%% 02 ---------------------------------------------------------- label distribution
fig = newFig('Optimal allocation distribution', [60 60 1150 450]);
tl  = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax  = nexttile(tl);
cnt = accumarray(data.labelIdx, 1, [numel(cand) 1]);
bar(ax, cand, 100 * cnt / sum(cnt), 0.8, 'FaceColor', [0.2 0.45 0.75]);
styleAxes(ax, 'Optimal N_{OFDM} (subcarriers)  [N_{OTFS} = 128 - N_{OFDM}]', 'Share of samples (%)', ...
          sprintf('Exhaustive-search labels (%d samples)', numel(data.labelIdx)));
xticks(ax, cand(1:2:end));
ax  = nexttile(tl); hold(ax, 'on');
edges = [0 500 1500 Inf];  lbl = {'f_D < 500 Hz', '500-1500 Hz', 'f_D > 1500 Hz'};
snrs  = unique(ds.meta.snr_dB).';  nOpt = cand(data.labelIdx).';
for b = 1:3
    m = ds.meta.fDmax >= edges(b) & ds.meta.fDmax < edges(b + 1) & ds.meta.dynFraction > 0;
    y = arrayfun(@(s) mean(nOpt(m & ds.meta.snr_dB == s)), snrs);
    plot(ax, snrs, y, '-o', 'LineWidth', 1.6, 'MarkerSize', 6);
end
y = arrayfun(@(s) mean(nOpt(ds.meta.dynFraction == 0 & ds.meta.snr_dB == s)), snrs);
plot(ax, snrs, y, 'k--s', 'LineWidth', 1.6, 'MarkerSize', 6);
styleAxes(ax, 'SNR (dB)', 'Mean optimal N_{OFDM} (subcarriers)', 'Optimal OFDM share vs SNR and Doppler');
legend(ax, [lbl, {'static channel (no moving paths)'}], 'Location', 'best');
ylim(ax, [0 K]);
saveFigure(fig, cfg, '02_optimal_allocation_distribution');

%% 03 / 04 ---------------------------------------------------------- training curves
h = model.history;
fig = newFig('Training and validation loss', [60 60 900 480]);
ax = axes(fig); hold(ax, 'on');
plot(ax, h.iteration, h.trainLoss, 'Color', [0.6 0.75 0.95], 'LineWidth', 0.8);
plot(ax, h.iteration, movmean(h.trainLoss, max(5, round(numel(h.iteration) / 50))), 'Color', [0 0.3 0.8], 'LineWidth', 1.8);
plot(ax, h.valIteration, h.valLoss, 'r-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
lossName = ternary(strcmp(model.mode, 'classification'), 'Cross-entropy loss', 'MSE loss (normalised N_{OFDM})');
styleAxes(ax, 'Iteration (mini-batch)', lossName, 'Training and validation loss');
legend(ax, {'training (per mini-batch)', 'training (moving average)', 'validation'}, 'Location', 'northeast');
saveFigure(fig, cfg, '03_training_validation_loss');

fig = newFig('Validation performance', [60 60 900 480]);
ax = axes(fig);
plot(ax, h.valIteration, h.valMetric, 'r-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
styleAxes(ax, 'Iteration (mini-batch)', h.metricName, 'Validation performance during training');
saveFigure(fig, cfg, '04_validation_performance');

%% 05 ---------------------------------------------------------- predicted vs actual
A  = numel(cand);
C  = accumarray([results.optimal.idx, results.dnn.idx], 1, [A A]);
Cn = 100 * C ./ max(sum(C, 2), 1);
fig = newFig('Predicted vs actual', [60 60 760 660]);
ax = axes(fig);
imagesc(ax, cand, cand, Cn, [0 100]);  axis(ax, 'xy');  axis(ax, 'square');
colormap(ax, [linspace(1, 0.05, 256).', linspace(1, 0.3, 256).', linspace(1, 0.65, 256).']);
cb = colorbar(ax);  cb.Label.String = 'Share of true class (%)';
hold(ax, 'on');  plot(ax, [0 K], [0 K], 'r--', 'LineWidth', 1.2);
styleAxes(ax, 'Predicted N_{OFDM} (subcarriers)', 'Optimal N_{OFDM} (subcarriers)', ...
          sprintf('DNN vs exhaustive search (exact %.1f%%, within \\pm%d: %.1f%%)', ...
          summary.exactMatch_pct, cfg.alloc.step, summary.within1_pct));
xticks(ax, cand(1:2:end));  yticks(ax, cand(1:2:end));
saveFigure(fig, cfg, '05_predicted_vs_actual');

%% 06 ---------------------------------------------------------- allocation error
fig = newFig('Allocation error', [60 60 820 460]);
ax = axes(fig);
st = cfg.alloc.step;
histogram(ax, summary.allocError, (-K - st/2):st:(K + st/2), 'Normalization', 'percentage', ...
          'FaceColor', [0.85 0.33 0.1]);
styleAxes(ax, 'Allocation error N_{OFDM}^{DNN} - N_{OFDM}^{opt} (subcarriers)', 'Share of test samples (%)', ...
          sprintf('Allocation error (MAE %.1f, RMSE %.1f subcarriers)', summary.MAE_subcarriers, summary.RMSE_subcarriers));
saveFigure(fig, cfg, '06_allocation_error_histogram');

%% 07 / 08 ---------------------------------------------------------- BER & SE vs SNR
bs = summary.bySNR;
for kind = {'BER', 'SE'}
    fig = newFig([kind{1} ' vs SNR'], [60 60 820 520]);
    ax = axes(fig); hold(ax, 'on');
    for r = sty.order
        plot(ax, bs.snr_dB, bs.(r{1}).(kind{1}), sty.(r{1}).ls, 'Color', sty.(r{1}).col, ...
             'LineWidth', sty.(r{1}).lw, 'MarkerSize', 7, 'DisplayName', sty.(r{1}).name);
    end
    if strcmp(kind{1}, 'BER')
        set(ax, 'YScale', 'log');
        styleAxes(ax, 'SNR per resource element (dB)', 'Uncoded BER (bit-weighted, both waveforms)', ...
                  sprintf('BER vs SNR (%d-QAM, test set)', cfg.sys.qamOrder));
        name = '07_ber_vs_snr';
    else
        styleAxes(ax, 'SNR per resource element (dB)', 'Spectral efficiency (bit/s/Hz)', ...
                  'Spectral efficiency vs SNR (test set, incl. pilot + CP overhead)');
        name = '08_spectral_efficiency_vs_snr';
    end
    legend(ax, 'Location', 'best');
    saveFigure(fig, cfg, name);
end

%% 09 ---------------------------------------------------------- DNN vs exhaustive
fig = newFig('DNN vs exhaustive search', [60 60 1300 440]);
tl  = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
for r = {'optimal', 'dnn', 'bestFixed', 'allOTFS'}
    plot(ax, bs.snr_dB, bs.(r{1}).score, sty.(r{1}).ls, 'Color', sty.(r{1}).col, 'LineWidth', sty.(r{1}).lw, ...
         'MarkerSize', 7, 'DisplayName', sty.(r{1}).name);
end
styleAxes(ax, 'SNR (dB)', 'Mean objective score', 'Objective score vs SNR');  legend(ax, 'Location', 'best');
ax = nexttile(tl); hold(ax, 'on');
for r = {'dnn', 'bestFixed', 'allOTFS', 'allOFDM'}
    reg = sort(results.optimal.score - results.(r{1}).score);
    plot(ax, reg, (1:numel(reg)) / numel(reg), regexprep(sty.(r{1}).ls, '[osd^vx]', ''), 'Color', sty.(r{1}).col, ...
         'LineWidth', 1.8, 'DisplayName', sty.(r{1}).name);
end
styleAxes(ax, 'Score regret  (score_{opt} - score)', 'Empirical CDF', 'Regret distribution (test set)');
legend(ax, 'Location', 'southeast');  xlim(ax, [0 max(0.3, pctl(results.optimal.score - results.allOFDM.score, 95))]);
ax = nexttile(tl);
names = sty.order;  pct = cellfun(@(r) summary.rules.(r).scorePctOfOpt, names);
b = bar(ax, categorical(cellfun(@(r) sty.(r).short, names, 'UniformOutput', false), ...
        cellfun(@(r) sty.(r).short, names, 'UniformOutput', false)), pct, 'FaceColor', 'flat');
for i = 1:numel(names), b.CData(i, :) = sty.(names{i}).col; end
text(ax, 1:numel(pct), pct, compose('%.1f%%', pct), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
styleAxes(ax, '', 'Mean score (% of optimal)', 'Performance relative to exhaustive search');
ylim(ax, [min(0, min(pct)) 110]);
saveFigure(fig, cfg, '09_dnn_vs_exhaustive');

%% 10 ---------------------------------------------------------- computation time
fig = newFig('Computation time', [60 60 820 480]);
ax  = axes(fig);
lab = {'Exhaustive search', 'DNN total', 'DNN features', 'DNN inference', 'DNN batched/sample'};
val = [timing.exhaustive_ms, timing.dnnTotal_ms, timing.dnnFeatures_ms, timing.dnnInference_ms, timing.dnnBatchPerSample_ms];
b = bar(ax, categorical(lab, lab), val, 'FaceColor', 'flat');
b.CData = [0.3 0.3 0.3; 0.85 0.15 0.15; 0.95 0.6 0.6; 0.95 0.6 0.6; 0.95 0.75 0.75];
set(ax, 'YScale', 'log');
text(ax, 1:numel(val), val, compose('%.3g ms', val), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
styleAxes(ax, '', 'Time per allocation decision (ms, log scale)', ...
          sprintf('Computation time (speed-up %.0fx, mean of %d decisions)', timing.speedup, timing.numSamples));
saveFigure(fig, cfg, '10_inference_time_comparison');

%% 11 ---------------------------------------------------------- trade-off
fig = newFig('Performance vs time', [60 60 860 520]);
ax  = axes(fig); hold(ax, 'on');
tr  = timing.tradeoff;
tMin = min([tr.time_ms]) / 10;
for i = 1:numel(tr)
    t = tr(i).time_ms;  if isnan(t), t = tMin; end
    if startsWith(tr(i).name, 'DNN'), c = sty.dnn.col; mk = 'p'; ms = 16;
    elseif startsWith(tr(i).name, 'Best'), c = sty.bestFixed.col; mk = 'd'; ms = 10;
    else, c = [0.3 0.3 0.3]; mk = 'o'; ms = 9; end
    plot(ax, t, tr(i).pctOfOpt, mk, 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', ms, 'DisplayName', tr(i).name);
end
srch = startsWith({tr.name}, 'Search');
plot(ax, [tr(srch).time_ms], [tr(srch).pctOfOpt], '-', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
set(ax, 'XScale', 'log');
styleAxes(ax, 'Computation time per decision (ms, log scale; fixed split plotted at left edge)', ...
          'Mean score (% of full exhaustive search)', 'Performance vs computation time');
legend(ax, 'Location', 'southeast');
saveFigure(fig, cfg, '11_performance_vs_time_tradeoff');

%% 12 ---------------------------------------------------------- sensing
fig = newFig('Detection probability', [60 60 820 500]);
ax  = axes(fig); hold(ax, 'on');
tg  = results.meta.tgtSNR_dB;
be  = pctl(tg, 0:10:100);  bc = (be(1:end-1) + be(2:end)) / 2;
for r = sty.order
    y = arrayfun(@(i) mean(results.(r{1}).Pd(tg >= be(i) & tg <= be(i + 1))), 1:numel(bc));
    plot(ax, bc, y, sty.(r{1}).ls, 'Color', sty.(r{1}).col, 'LineWidth', sty.(r{1}).lw, 'MarkerSize', 7, ...
         'DisplayName', sty.(r{1}).name);
end
styleAxes(ax, 'Target echo SNR per resource element (dB)', sprintf('Detection probability (P_{fa} = %g)', cfg.sens.Pfa), ...
          'Sensing: detection probability of the target (OTFS sub-band radar)');
legend(ax, 'Location', 'northwest');  ylim(ax, [0 1]);
saveFigure(fig, cfg, '12_detection_probability');

%% 13 ---------------------------------------------------------- allocation vs Doppler
fig = newFig('Allocation vs Doppler', [60 60 820 480]);
ax  = axes(fig); hold(ax, 'on');
fd  = results.meta.fDmax .* (results.meta.dynFraction > 0);
fe  = linspace(0, cfg.chan.maxDopplerRange(2), 9);  fcn = (fe(1:end-1) + fe(2:end)) / 2;
yo  = arrayfun(@(i) mean(results.optimal.nOFDM(fd >= fe(i) & fd < fe(i + 1))), 1:numel(fcn));
yd  = arrayfun(@(i) mean(results.dnn.nOFDM(fd >= fe(i) & fd < fe(i + 1))), 1:numel(fcn));
plot(ax, fcn, yo, sty.optimal.ls, 'Color', sty.optimal.col, 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Optimal (exhaustive)');
plot(ax, fcn, yd, sty.dnn.ls, 'Color', sty.dnn.col, 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'DNN prediction');
styleAxes(ax, 'Maximum Doppler of moving paths f_D (Hz)', 'Mean N_{OFDM} (subcarriers)', ...
          'Learned trend: OFDM share vs mobility (test set)');
legend(ax, 'Location', 'best');  ylim(ax, [0 K]);
saveFigure(fig, cfg, '13_allocation_vs_doppler');

fprintf('Saved figures to %s\n', cfg.paths.plots);
end

% ==============================================================================
function fig = newFig(name, pos)
fig = figure('Name', name, 'Color', 'w', 'Position', pos);
end

function sty = ruleStyles(bestFixedNofdm)
sty.order = {'optimal', 'dnn', 'bestFixed', 'allOTFS', 'allOFDM', 'half'};
sty.optimal   = struct('name', 'Optimal (exhaustive search)', 'short', 'Optimal', 'col', [0 0 0],          'ls', '-o',  'lw', 2.2);
sty.dnn       = struct('name', 'DNN prediction',              'short', 'DNN',     'col', [0.85 0.1 0.1],    'ls', '--s', 'lw', 2.0);
sty.bestFixed = struct('name', sprintf('Best fixed split (N_{OFDM}=%d)', bestFixedNofdm), 'short', 'Best fixed', ...
                                                                                'col', [0.49 0.18 0.56], 'ls', ':d',  'lw', 1.6);
sty.allOTFS   = struct('name', 'All OTFS (N_{OFDM}=0)',       'short', 'All OTFS', 'col', [0 0.45 0.74],    'ls', '-.^', 'lw', 1.4);
sty.allOFDM   = struct('name', 'All OFDM (N_{OFDM}=128)',     'short', 'All OFDM', 'col', [0.47 0.67 0.19], 'ls', '-.v', 'lw', 1.4);
sty.half      = struct('name', 'Half/half (64/64)',           'short', '64/64',    'col', [0.6 0.6 0.6],    'ls', ':x',  'lw', 1.4);
end

function q = pctl(x, p)
% Percentiles without Statistics and Machine Learning Toolbox (nearest rank).
x = sort(x(:));
q = x(max(1, min(numel(x), round(p / 100 * numel(x)))));
q = reshape(q, size(p));
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
