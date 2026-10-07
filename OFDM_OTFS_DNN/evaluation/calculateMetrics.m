function summary = calculateMetrics(results, cfg)
%CALCULATEMETRICS Allocation accuracy and performance gaps: DNN and baselines vs. optimum.
%
%   summary = calculateMetrics(results, cfg)
%
%   Allocation accuracy (DNN vs exhaustive-search label):
%     exactMatch_pct   share of test samples with N_OFDM_pred == N_OFDM_opt
%     within1_pct      |error| <= 1 step (e.g. 8 subcarriers)
%     MAE, RMSE        allocation error in subcarriers
%   Performance (every rule vs the optimum, averaged over the test set):
%     scoreRegret      mean(score_opt - score_rule) >= 0
%     scorePctOfOpt    100 * mean(score_rule) / mean(score_opt)
%     SE, dSE_pct      mean spectral efficiency and its relative loss
%     BER, dBER        mean bit-weighted BER and its difference
%     Pd, dPd          mean detection probability and its difference
%   Accuracy alone is misleading here: several splits are often almost
%   equally good, so a "wrong" class can cost nothing.  The regret and the
%   SE/BER/Pd gaps are the quantities that matter.
%
%   Also returns per-SNR tables (summary.bySNR) used by the plots.

opt  = results.optimal;
step = cfg.alloc.step;
err  = results.dnn.nOFDM - opt.nOFDM;

summary.exactMatch_pct = 100 * mean(err == 0);
summary.within1_pct    = 100 * mean(abs(err) <= step);
summary.within2_pct    = 100 * mean(abs(err) <= 2 * step);
summary.MAE_subcarriers  = mean(abs(err));
summary.RMSE_subcarriers = sqrt(mean(err.^2));
summary.allocError       = err;

for r = results.ruleNames
    R = results.(r{1});
    s.score         = mean(R.score);
    s.scoreRegret   = mean(opt.score - R.score);
    s.scorePctOfOpt = 100 * mean(R.score) / mean(opt.score);
    s.SE    = mean(R.SE);   s.dSE_pct = 100 * (mean(R.SE) - mean(opt.SE)) / mean(opt.SE);
    s.BER   = mean(R.BER);  s.dBER    = mean(R.BER) - mean(opt.BER);
    s.Pd    = mean(R.Pd);   s.dPd     = mean(R.Pd) - mean(opt.Pd);
    s.exactMatch_pct = 100 * mean(R.idx == opt.idx);
    summary.rules.(r{1}) = s;
end

% ------------------------------------------------------------ per-SNR breakdown
snrs = unique(results.snr_dB).';
summary.bySNR.snr_dB = snrs;
for r = results.ruleNames
    R = results.(r{1});
    for i = 1:numel(snrs)
        m = results.snr_dB == snrs(i);
        summary.bySNR.(r{1}).SE(i)    = mean(R.SE(m));
        summary.bySNR.(r{1}).BER(i)   = mean(R.BER(m));
        summary.bySNR.(r{1}).Pd(i)    = mean(R.Pd(m));
        summary.bySNR.(r{1}).score(i) = mean(R.score(m));
        summary.bySNR.(r{1}).nOFDM(i) = mean(R.nOFDM(m));
    end
end
for i = 1:numel(snrs)
    m = results.snr_dB == snrs(i);
    summary.bySNR.exactMatch_pct(i) = 100 * mean(err(m) == 0);
end

% ------------------------------------------------------------ console report
fprintf('\n=================== TEST-SET RESULTS (%d samples) ===================\n', results.numTest);
fprintf('Allocation accuracy (DNN vs exhaustive search):\n');
fprintf('  exact match            : %6.2f %%\n', summary.exactMatch_pct);
fprintf('  within +-%d subcarriers : %6.2f %%\n', step, summary.within1_pct);
fprintf('  within +-%d subcarriers: %6.2f %%\n', 2 * step, summary.within2_pct);
fprintf('  MAE / RMSE             : %.2f / %.2f subcarriers\n\n', summary.MAE_subcarriers, summary.RMSE_subcarriers);
fprintf('%-10s %8s %9s %9s %8s %10s %7s %8s\n', 'Rule', 'Score', '%ofOpt', 'SE', 'dSE%', 'BER', 'Pd', 'dPd');
for r = results.ruleNames
    s = summary.rules.(r{1});
    fprintf('%-10s %8.4f %8.2f%% %9.3f %+7.2f%% %10.2e %7.3f %+8.3f\n', r{1}, s.score, ...
            s.scorePctOfOpt, s.SE, s.dSE_pct, s.BER, s.Pd, s.dPd);
end
fprintf('(bestFixed = N_OFDM %d, chosen on the training set)\n', results.bestFixedNofdm);
fprintf('=====================================================================\n');
end
