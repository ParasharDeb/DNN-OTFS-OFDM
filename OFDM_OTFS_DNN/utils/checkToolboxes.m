function tb = checkToolboxes()
%CHECKTOOLBOXES Report which MathWorks products are available and stop if a required one is missing.
%
%   Required : Deep Learning Toolbox          (featureInputLayer, trainnet / trainNetwork)
%   Recommended: Communications Toolbox      (qammod/qamdemod for the waveform validation,
%                                             berawgn for BER tables; closed-form fallback exists)
%   Optional : 5G Toolbox                     (3GPP TDL power-delay profiles via nrTDLChannel;
%                                             exponential-PDP fallback exists)
%              Parallel Computing Toolbox     (parfor dataset generation)
%   Not needed: Statistics & ML Toolbox, Signal Processing Toolbox, Phased Array
%              System Toolbox (radar equation is implemented in closed form).

tb.deepLearning = ~isempty(which('featureInputLayer'));
tb.trainnet     = ~isempty(which('trainnet'));
tb.comm         = ~isempty(which('qammod')) && ~isempty(which('berawgn'));
tb.fiveG        = ~isempty(which('nrTDLChannel'));
tb.parallel     = ~isempty(ver('parallel'));

fprintf('MATLAB %s\n', version);
fprintf('  Deep Learning Toolbox      : %s%s\n', yn(tb.deepLearning), ternary(tb.trainnet, ' (trainnet API)', ' (trainNetwork API)'));
fprintf('  Communications Toolbox     : %s\n', yn(tb.comm));
fprintf('  5G Toolbox (TDL profiles)  : %s\n', yn(tb.fiveG));
fprintf('  Parallel Computing Toolbox : %s\n', yn(tb.parallel));

if ~tb.deepLearning
    error('Deep Learning Toolbox (R2020b or newer, featureInputLayer) is required.');
end
if exist('isMATLABReleaseOlderThan') == 0 || isMATLABReleaseOlderThan('R2021a') %#ok<EXIST>
    warning('This project was written for MATLAB R2021a or newer.');
end
end

function s = yn(b)
s = ternary(b, 'yes', 'NO');
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
