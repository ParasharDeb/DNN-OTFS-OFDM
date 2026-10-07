function [nOFDM, nOTFS, scores, elapsed_s] = predictAllocation(model, Xraw)
%PREDICTALLOCATION DNN inference: raw feature vector(s) -> valid OFDM/OTFS split.
%
%   [nOFDM, nOTFS, scores, elapsed_s] = predictAllocation(model, Xraw)
%
%   Xraw   : [n x F] UN-normalised features (output of extractFeatures)
%   nOFDM  : [n x 1] predicted OFDM subcarriers, always one of model.candidates
%   nOTFS  : [n x 1] = N_total - nOFDM   (the constraint holds by construction)
%   scores : [n x A] softmax class probabilities (classification) or
%            [n x 1] raw regression output N_OFDM/N_total (regression)
%   elapsed_s : wall-clock time of normalisation + forward pass + decision

arguments
    model struct
    Xraw {mustBeNumeric, mustBeReal}
end
assert(size(Xraw, 2) == numel(model.norm.mu), ...
       'predictAllocation: expected %d features, got %d.', numel(model.norm.mu), size(Xraw, 2));

t0 = tic;
X  = (Xraw - model.norm.mu) ./ model.norm.sigma;

if model.useTrainnet
    Y = predict(model.net, dlarray(single(X).', 'CB'));
    Y = double(extractdata(Y)).';
else
    Y = double(predict(model.net, X));
end

switch model.mode
    case 'classification'
        scores = Y;
        [~, k] = max(scores, [], 2);
        nOFDM  = model.candidates(k).';
    case 'regression'
        scores = Y;
        % Map continuous output to the nearest allowed allocation and clip to [0, N_total].
        nOFDM  = round(Y * model.N_total / model.step) * model.step;
        nOFDM  = min(max(nOFDM, 0), model.N_total);
end
nOFDM = nOFDM(:);
nOTFS = model.N_total - nOFDM;
elapsed_s = toc(t0);
end
