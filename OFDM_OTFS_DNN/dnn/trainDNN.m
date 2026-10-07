function model = trainDNN(cfg, data)
%TRAINDNN Train the allocation DNN with Adam, LR schedule and early stopping.
%
%   model = trainDNN(cfg, data)        data from preprocessDataset
%
%   * Optimiser: Adam, mini-batches of cfg.dnn.miniBatchSize, shuffled every epoch
%   * Learning rate: piecewise schedule (x dropFactor every dropPeriod epochs)
%   * L2 weight decay + dropout against over-fitting
%   * Early stopping: training stops when the validation loss has not improved
%     for cfg.dnn.validationPatience validation checks; the best-validation
%     network is returned where the MATLAB release supports it
%   * Reproducible: fixed seed + CPU execution (GPU kernels are not
%     bit-deterministic)
%
%   Output struct 'model' (everything needed for inference, saved by main):
%     net, mode, useTrainnet, norm, candidates, N_total, featureNames,
%     history.{iteration, trainLoss, valIteration, valLoss, valMetric, metricName},
%     trainingTime_s

rng(cfg.seed, 'twister');
F = size(data.XTrain, 2);
A = numel(data.candidates);
[layers, useTrainnet] = buildDNN(cfg, F, A);

itPerEpoch = max(1, floor(size(data.XTrain, 1) / cfg.dnn.miniBatchSize));
base = {'MaxEpochs', cfg.dnn.maxEpochs, ...
        'MiniBatchSize', cfg.dnn.miniBatchSize, ...
        'InitialLearnRate', cfg.dnn.initialLearnRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', cfg.dnn.learnRateDropPeriod, ...
        'LearnRateDropFactor', cfg.dnn.learnRateDropFactor, ...
        'L2Regularization', cfg.dnn.l2Regularization, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {data.XVal, data.YVal}, ...
        'ValidationFrequency', itPerEpoch * cfg.dnn.validationFrequencyEpochs, ...
        'ValidationPatience', cfg.dnn.validationPatience, ...
        'ExecutionEnvironment', 'cpu', ...
        'Verbose', true, 'VerboseFrequency', 5 * itPerEpoch, ...
        'Plots', ternary(cfg.dnn.showTrainingPlot, 'training-progress', 'none')};

isCls = strcmp(cfg.dnn.mode, 'classification');
fprintf('Training %s DNN (%s), %d features -> %d classes, %d training samples...\n', ...
        cfg.dnn.mode, ternary(useTrainnet, 'trainnet', 'trainNetwork'), F, A, size(data.XTrain, 1));

t0 = tic;
if useTrainnet
    metric = ternary(isCls, 'accuracy', 'rmse');
    opts = tryOptions(base, {'Metrics', metric, 'OutputNetwork', 'best-validation'}, ...
                            {'Metrics', metric});
    lossFcn = ternary(isCls, 'crossentropy', 'mse');
    [net, info] = trainnet(data.XTrain, data.YTrain, layers, lossFcn, opts);
    hist = historyFromTrainnet(info, isCls, cfg.sys.N_total);
else
    opts = tryOptions(base, {'OutputNetwork', 'best-validation-loss'}, {});
    [net, info] = trainNetwork(data.XTrain, data.YTrain, layers, opts);
    hist = historyFromTrainNetwork(info, isCls, cfg.sys.N_total);
end
model.trainingTime_s = toc(t0);

model.net          = net;
model.mode         = cfg.dnn.mode;
model.useTrainnet  = useTrainnet;
model.norm         = data.norm;
model.candidates   = data.candidates;
model.N_total      = cfg.sys.N_total;
model.step         = cfg.alloc.step;
model.featureNames = data.featureNames;
model.history      = hist;
fprintf('Training finished in %.1f s.\n', model.trainingTime_s);
end

% ==============================================================================
function opts = tryOptions(base, preferred, fallback)
% Some options (Metrics, OutputNetwork values) depend on the MATLAB release.
try
    opts = trainingOptions('adam', base{:}, preferred{:});
catch
    opts = trainingOptions('adam', base{:}, fallback{:});
end
end

function h = historyFromTrainNetwork(info, isCls, Ntot)
h.iteration = 1:numel(info.TrainingLoss);
h.trainLoss = info.TrainingLoss;
v = find(~isnan(info.ValidationLoss));
h.valIteration = v;
h.valLoss      = info.ValidationLoss(v);
if isCls
    h.valMetric = info.ValidationAccuracy(v);   h.metricName = 'Validation accuracy (%)';
else
    h.valMetric = info.ValidationRMSE(v) * Ntot; h.metricName = 'Validation RMSE of N_{OFDM} (subcarriers)';
end
end

function h = historyFromTrainnet(info, isCls, Ntot)
th = info.TrainingHistory;
vh = info.ValidationHistory;
h.iteration    = th.Iteration.';
h.trainLoss    = th.TrainingLoss.';
h.valIteration = vh.Iteration.';
h.valLoss      = vh.ValidationLoss.';
vn = vh.Properties.VariableNames;
if isCls && ismember('ValidationAccuracy', vn)
    h.valMetric = vh.ValidationAccuracy.';  h.metricName = 'Validation accuracy (%)';
elseif ~isCls && ismember('ValidationRMSE', vn)
    h.valMetric = vh.ValidationRMSE.' * Ntot;  h.metricName = 'Validation RMSE of N_{OFDM} (subcarriers)';
else
    h.valMetric = h.valLoss;  h.metricName = 'Validation loss';
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
