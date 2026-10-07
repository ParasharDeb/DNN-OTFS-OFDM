function [layers, useTrainnet] = buildDNN(cfg, numFeatures, numClasses)
%BUILDDNN Fully connected DNN: features -> optimal OFDM/OTFS split.
%
%   [layers, useTrainnet] = buildDNN(cfg, numFeatures, numClasses)
%
%   Architecture (cfg.dnn.hiddenUnits = [128 64 32] by default):
%       featureInput(F) -> [FC(h) -> ReLU -> Dropout] x numel(hiddenUnits) -> head
%   Heads
%     classification : FC(numClasses) -> softmax            (default)
%                      predicts one of the 17 allowed splits 0/128, 8/120, ...
%     regression     : FC(1)  (sigmoid-free; output clipped/rounded later)
%                      predicts N_OFDM / N_total
%
%   Why classification is the default (README, "Regression vs classification"):
%     * The label set is small and discrete (17 values); every prediction is
%       a valid allocation by construction, no rounding/clipping heuristics.
%     * The optimal split is often bimodal for a given feature vector (e.g.
%       "all-OTFS" vs "small OTFS band" have similar scores).  An MSE
%       regressor then predicts the mean of the two modes, an allocation that
%       may be poor for both.  Softmax keeps both modes and picks the larger.
%     * Softmax scores give a confidence measure that can trigger a fallback
%       to the exhaustive search when the DNN is unsure.
%   Drawback: cross-entropy ignores ordinal distance (predicting 120 for a
%   true 0 costs the same as predicting 8).  We therefore always report the
%   performance regret, not only exact-match accuracy.
%
%   Uses trainnet-style layers (no output layer) on R2023b+, otherwise
%   classic trainNetwork layers with classificationLayer/regressionLayer.

arguments
    cfg struct
    numFeatures (1,1) {mustBeInteger, mustBePositive}
    numClasses  (1,1) {mustBeInteger, mustBePositive}
end

useTrainnet = ~isempty(which('trainnet'));

layers = featureInputLayer(numFeatures, 'Name', 'csi_features', 'Normalization', 'none');
for i = 1:numel(cfg.dnn.hiddenUnits)
    layers = [layers
              fullyConnectedLayer(cfg.dnn.hiddenUnits(i), 'Name', sprintf('fc%d', i))
              reluLayer('Name', sprintf('relu%d', i))];                         %#ok<AGROW>
    if cfg.dnn.dropout > 0
        layers = [layers; dropoutLayer(cfg.dnn.dropout, 'Name', sprintf('drop%d', i))]; %#ok<AGROW>
    end
end

switch cfg.dnn.mode
    case 'classification'
        layers = [layers
                  fullyConnectedLayer(numClasses, 'Name', 'fc_out')
                  softmaxLayer('Name', 'softmax')];
        if ~useTrainnet
            layers = [layers; classificationLayer('Name', 'output')];
        end
    case 'regression'
        layers = [layers; fullyConnectedLayer(1, 'Name', 'fc_out')];
        if ~useTrainnet
            layers = [layers; regressionLayer('Name', 'output')];
        end
    otherwise
        error('buildDNN: unknown mode "%s".', cfg.dnn.mode);
end
end
