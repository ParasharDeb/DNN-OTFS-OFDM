function data = preprocessDataset(ds, cfg)
%PREPROCESSDATASET Labels (for the current weights), train/val/test split, normalisation.
%
%   data = preprocessDataset(ds, cfg)
%
%   1. Re-labelling: labels are recomputed from the stored per-candidate
%      metrics with the CURRENT cfg.obj weights (identical to generation-time
%      labels if the weights did not change).
%   2. Split 70/15/15 (cfg.data.split), stratified by label so that rare
%      allocations appear in every subset.  Every sample is an independent
%      channel realisation with its own seed, so no channel is shared between
%      subsets (no leakage through repeated channels).
%   3. Z-score normalisation with mean/std estimated on the TRAINING set only
%      and applied unchanged to validation/test (no test-statistics leakage).
%
%   Output struct fields:
%     XTrain/XVal/XTest       normalised features [n x F]
%     YTrain/YVal/YTest       categorical (classification) or N_OFDM/N_total (regression)
%     idxTrain/idxVal/idxTest row indices into ds
%     labelIdx                [nS x 1] optimal class index for every sample
%     norm.mu, norm.sigma     normalisation statistics (1 x F)
%     classNames, candidates, featureNames, mode

assert(size(ds.X, 1) == numel(ds.meta.snr_dB), 'preprocessDataset: inconsistent dataset.');
nS = size(ds.X, 1);
A  = numel(ds.candidates);

% ---------------------------------------------------------------- 1. labels
score = computeScore(ds.metrics.SE, ds.metrics.BER, ds.metrics.Pd, ds.meta.snr_dB, cfg.obj, ...
                     (ds.N_total - ds.candidates) / ds.N_total);
[~, labelIdx] = max(score, [], 2);
if ~isequal(labelIdx, ds.labelIdx)
    fprintf('  Objective weights differ from generation time: %d of %d labels changed.\n', ...
            nnz(labelIdx ~= ds.labelIdx), nS);
end

% ---------------------------------------------------------------- 2. stratified split
rng(cfg.seed, 'twister');
sp = cfg.data.split;
[idxTr, idxVa, idxTe] = deal([]);
for c = 1:A
    ic = find(labelIdx == c);
    ic = ic(randperm(numel(ic)));
    nTr = round(sp(1) * numel(ic));
    nVa = round(sp(2) * numel(ic));
    idxTr = [idxTr; ic(1:nTr)];                 %#ok<AGROW>
    idxVa = [idxVa; ic(nTr+1:nTr+nVa)];         %#ok<AGROW>
    idxTe = [idxTe; ic(nTr+nVa+1:end)];         %#ok<AGROW>
end
idxTr = idxTr(randperm(numel(idxTr)));
idxVa = sort(idxVa);  idxTe = sort(idxTe);
assert(isempty(intersect(idxTr, idxVa)) && isempty(intersect(idxTr, idxTe)) && ...
       isempty(intersect(idxVa, idxTe)), 'Split overlap detected.');
assert(numel(idxTr) + numel(idxVa) + numel(idxTe) == nS, 'Split lost samples.');

% ---------------------------------------------------------------- 3. normalisation
mu    = mean(ds.X(idxTr, :), 1);
sigma = std(ds.X(idxTr, :), 0, 1);
sigma(sigma < 1e-12) = 1;                       % constant feature -> leave centred only
nrm   = @(X) (X - mu) ./ sigma;

data.mode         = cfg.dnn.mode;
data.candidates   = ds.candidates;
data.classNames   = string(ds.candidates);
data.featureNames = ds.featureNames;
data.labelIdx     = labelIdx;
data.score        = score;
data.idxTrain = idxTr;  data.idxVal = idxVa;  data.idxTest = idxTe;
data.XTrain = nrm(ds.X(idxTr, :));
data.XVal   = nrm(ds.X(idxVa, :));
data.XTest  = nrm(ds.X(idxTe, :));
data.norm.mu = mu;  data.norm.sigma = sigma;

switch cfg.dnn.mode
    case 'classification'
        toCat = @(i) categorical(labelIdx(i), 1:A, data.classNames);
        data.YTrain = toCat(idxTr);  data.YVal = toCat(idxVa);  data.YTest = toCat(idxTe);
    case 'regression'
        % Target scaled to [0, 1]: N_OFDM / N_total
        toReg = @(i) ds.candidates(labelIdx(i)).' / ds.N_total;
        data.YTrain = toReg(idxTr);  data.YVal = toReg(idxVa);  data.YTest = toReg(idxTe);
end

fprintf('  Split: %d train / %d val / %d test, %d features, %d classes.\n', ...
        numel(idxTr), numel(idxVa), numel(idxTe), size(ds.X, 2), A);
end
