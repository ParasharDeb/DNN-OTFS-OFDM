function results = testDNN(cfg, model, ds, data)
%TESTDNN Apply the trained DNN to the held-out test set and score its decisions exactly.
%
%   results = testDNN(cfg, model, ds, data)
%
%   For every test sample the dataset holds the true metrics of ALL candidate
%   splits (computed by the exhaustive search with the true channel), so the
%   communication and sensing performance of the DNN's chosen split is a
%   table lookup, with no approximation and no re-simulation.
%
%   Compared decision rules (all evaluated on the same test channels):
%     optimal   : exhaustive-search label (genie, needs the full channel)
%     dnn       : DNN prediction from the imperfect observations
%     allOTFS   : fixed  N_OFDM = 0
%     allOFDM   : fixed  N_OFDM = N_total
%     half      : fixed  N_OFDM = N_total/2
%     bestFixed : the single fixed split with the highest mean score on the
%                 TRAINING set (the strongest non-adaptive baseline)
%
%   Output: results.(rule).{idx, nOFDM, SE, BER, BER_OFDM, BER_OTFS, Pd, score},
%           results.snr_dB, results.meta (test-sample metadata), results.dnnScores,
%           results.dnnBatchTime_s, results.ruleNames

idxT = data.idxTest;
nT   = numel(idxT);
A    = numel(ds.candidates);
cand = ds.candidates;

[nPred, ~, scores, tBatch] = predictAllocation(model, ds.X(idxT, :));
[~, predIdx] = ismember(nPred, cand);
assert(all(predIdx > 0), 'testDNN: prediction outside the candidate set.');

trainScoreMean = mean(data.score(data.idxTrain, :), 1);
[~, bestFixedIdx] = max(trainScoreMean);

rules.optimal   = data.labelIdx(idxT);
rules.dnn       = predIdx(:);
rules.allOTFS   = ones(nT, 1);
rules.allOFDM   = A * ones(nT, 1);
rules.half      = find(cand == cfg.sys.N_total / 2) * ones(nT, 1);
rules.bestFixed = bestFixedIdx * ones(nT, 1);

metricNames = {'SE', 'BER', 'BER_OFDM', 'BER_OTFS', 'SINR_OFDM_dB', 'SINR_OTFS_dB', 'Pd', 'sensSINR_dB'};
rn = fieldnames(rules);
for r = rn.'
    k = rules.(r{1});
    lin = sub2ind([size(ds.X, 1), A], idxT(:), k);
    results.(r{1}).idx   = k;
    results.(r{1}).nOFDM = cand(k).';
    for m = metricNames
        results.(r{1}).(m{1}) = ds.metrics.(m{1})(lin);
    end
    results.(r{1}).score = data.score(lin);
end

results.ruleNames      = rn.';
results.bestFixedNofdm = cand(bestFixedIdx);
results.snr_dB         = ds.meta.snr_dB(idxT);
mf = fieldnames(ds.meta);
for f = mf.'
    results.meta.(f{1}) = ds.meta.(f{1})(idxT);
end
results.dnnScores      = scores;
results.dnnBatchTime_s = tBatch;
results.numTest        = nT;
results.candidates     = cand;
results.N_total        = cfg.sys.N_total;

fprintf('Tested DNN on %d held-out samples (batch inference %.2f ms total).\n', nT, 1e3 * tBatch);
end
