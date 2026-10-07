%MAIN End-to-end pipeline: AI-based OFDM/OTFS resource allocation for ISAC.
%
%   >> main
%
%   Stage 0  Toolbox check, paths, seed
%   Stage 1  Link-abstraction validation (waveform-level Monte Carlo)   [optional]
%   Stage 2  Dataset: channel -> CSI -> exhaustive search -> labels    [cached]
%   Stage 3  Pre-processing: labels for current weights, split, normalisation
%   Stage 4  DNN training (or reuse of the saved model)
%   Stage 5  Testing on held-out channels: DNN vs exhaustive search vs fixed splits
%   Stage 6  Computation-time comparison and accuracy/time trade-off
%   Stage 7  Figures
%   Stage 8  Save model + results, print the final demonstration
%
%   All parameters: config/simulationConfig.m.  Expected run time with the
%   default 12 000 samples: dataset ~15-40 min (parfor) / 1-3 h (serial),
%   training ~1-3 min, everything else ~2 min.  Set cfg.data.numSamples =
%   1500 for a quick smoke test.

clear; close all; clc;
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'config'), fullfile(root, 'simulation'), fullfile(root, 'optimization'), ...
        fullfile(root, 'dataset'), fullfile(root, 'dnn'), fullfile(root, 'evaluation'), fullfile(root, 'utils'));

%% Stage 0 ------------------------------------------------------------------------
cfg = simulationConfig();
tb  = checkToolboxes();
cfg.paths.results = fullfile(root, 'results');
for d = {cfg.paths.data, cfg.paths.plots, cfg.paths.models, cfg.paths.results}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end
rng(cfg.seed, 'twister');
fprintf('\nTotal subcarriers: %d | candidates N_OFDM = %s | %d-QAM | weights a=%.2f b=%.2f g=%.2f d=%.2f\n\n', ...
        cfg.sys.N_total, mat2str(cfg.alloc.candidates), cfg.sys.qamOrder, ...
        cfg.obj.alpha, cfg.obj.beta, cfg.obj.gamma, cfg.obj.delta);

%% Stage 1 ------------------------------------------------------------------------
if cfg.validation.enable
    if tb.comm
        fprintf('=== Stage 1: link-abstraction validation ===\n');
        val = validateLinkAbstraction(cfg);
        save(fullfile(cfg.paths.results, 'linkValidation.mat'), 'val');
    else
        warning('Communications Toolbox not found: skipping waveform-level validation.');
    end
end

%% Stage 2 ------------------------------------------------------------------------
fprintf('\n=== Stage 2: dataset ===\n');
dsFile = fullfile(cfg.paths.data, 'dataset.mat');
reuse  = false;
if ~cfg.run.regenerateDataset && exist(dsFile, 'file')
    S = load(dsFile, 'ds');
    reuse = isfield(S.ds, 'fingerprint') && strcmp(S.ds.fingerprint, datasetFingerprint(cfg));
    if reuse
        ds = S.ds;
        fprintf('Reusing %s (%d samples, created %s).\n', dsFile, size(ds.X, 1), ds.created);
    else
        fprintf('Existing dataset was generated with different settings: regenerating.\n');
    end
    clear S;
end
if ~reuse
    ds = generateDataset(cfg);
end

%% Stage 3 ------------------------------------------------------------------------
fprintf('\n=== Stage 3: pre-processing ===\n');
data = preprocessDataset(ds, cfg);

%% Stage 4 ------------------------------------------------------------------------
fprintf('\n=== Stage 4: DNN training ===\n');
modelFile = fullfile(cfg.paths.models, 'allocationDNN.mat');
if cfg.run.retrain || ~exist(modelFile, 'file')
    model = trainDNN(cfg, data);
    model.objUsed = cfg.obj;
    model.datasetFingerprint = ds.fingerprint;
    save(modelFile, 'model');
    fprintf('Saved trained model to %s\n', modelFile);
else
    S = load(modelFile, 'model');  model = S.model;  clear S;
    fprintf('Loaded trained model from %s\n', modelFile);
end

%% Stage 5 ------------------------------------------------------------------------
fprintf('\n=== Stage 5: testing ===\n');
results = testDNN(cfg, model, ds, data);
summary = calculateMetrics(results, cfg);

%% Stage 6 ------------------------------------------------------------------------
fprintf('\n=== Stage 6: computation time ===\n');
timing = compareWithBaseline(cfg, model, ds, data, results);

%% Stage 7 ------------------------------------------------------------------------
fprintf('\n=== Stage 7: figures ===\n');
plotResults(cfg, ds, data, model, results, summary, timing);

%% Stage 8 ------------------------------------------------------------------------
save(fullfile(cfg.paths.results, 'results.mat'), 'results', 'summary', 'timing', 'cfg');

% Final demonstration on one held-out test channel (re-simulated from its seed).
k   = min(cfg.eval.demoSampleIndex, numel(data.idxTest));
row = data.idxTest(k);
smp = regenerateSample(cfg, ds.meta.sampleId(row));
tE  = tic;  ex = exhaustiveSearch(cfg, smp.ch, smp.scene, smp.snr_dB);  tE = toc(tE);
tD  = tic;
x   = extractFeatures(cfg, smp.obs, smp.snr_dB);
[nO, nT] = predictAllocation(model, x);
tD  = toc(tD);
dnnIdx = find(cfg.alloc.candidates == nO);

fprintf('\n======================== FINAL DEMONSTRATION ========================\n');
fprintf('Test channel #%d: SNR %d dB, f_D %.0f Hz (%.0f%% moving paths), DS %.0f ns, %s\n', ...
        ds.meta.sampleId(row), smp.snr_dB, smp.ch.fDmax, 100 * smp.ch.dynFraction, ...
        1e9 * smp.ch.delaySpread, smp.ch.profile);
fprintf('Total subcarriers: %d\n\n', cfg.sys.N_total);
fprintf('Ground truth allocation (exhaustive search):\n  OFDM = %d\n  OTFS = %d\n\n', ex.bestNofdm, ex.bestNotfs);
fprintf('DNN prediction:\n  OFDM = %d\n  OTFS = %d\n\n', nO, nT);
fprintf('Allocation error: %d subcarriers\n\n', nO - ex.bestNofdm);
fprintf('Performance           optimal      DNN\n');
fprintf('  score             %8.4f  %8.4f\n', ex.score(ex.bestIdx), ex.score(dnnIdx));
fprintf('  SE (bit/s/Hz)     %8.3f  %8.3f\n', ex.SE(ex.bestIdx), ex.SE(dnnIdx));
fprintf('  BER               %8.2e  %8.2e\n', ex.BER(ex.bestIdx), ex.BER(dnnIdx));
fprintf('  Pd (sensing)      %8.3f  %8.3f\n\n', ex.Pd(ex.bestIdx), ex.Pd(dnnIdx));
fprintf('Inference time:\n  Exhaustive search = %.2f ms\n  DNN               = %.3f ms  (features + inference)\n', 1e3 * tE, 1e3 * tD);
fprintf('\nTest-set summary: exact match %.1f %%, within +-%d: %.1f %%, DNN reaches %.2f %% of the optimal score\n', ...
        summary.exactMatch_pct, cfg.alloc.step, summary.within1_pct, summary.rules.dnn.scorePctOfOpt);
fprintf('(best fixed split reaches %.2f %%), mean speed-up %.0fx.\n', summary.rules.bestFixed.scorePctOfOpt, timing.speedup);
fprintf('=====================================================================\n');
