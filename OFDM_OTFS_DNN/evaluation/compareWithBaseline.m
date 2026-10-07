function timing = compareWithBaseline(cfg, model, ds, data, results)
%COMPAREWITHBASELINE Computation time of exhaustive search vs. DNN, and the accuracy/time trade-off.
%
%   timing = compareWithBaseline(cfg, model, ds, data, results)
%
%   Timing is measured serially on cfg.eval.numTimingSamples test samples
%   (same machine, same MATLAB session, no parfor), after one warm-up call.
%
%   Exhaustive search per decision =
%       building the TF channel matrices (needs the TRUE delay-Doppler channel)
%     + evaluating all candidate splits (OFDM SINR, OTFS LMMSE, BER, Pd)
%   DNN per decision =
%       feature extraction from the observed CSI (extractFeatures)
%     + normalisation + forward pass + argmax (predictAllocation)
%   Not timed for either: the CSI acquisition itself (pilots, radar frame),
%   which both approaches need in some form.
%
%   Fairness note: the exhaustive search additionally needs information a
%   real system does not have (exact path delays/Dopplers/gains and the
%   sensing scene), so in practice it is an upper bound, not a deployable
%   alternative.  The DNN is compared against it for performance, and
%   against cheaper searches (coarser candidate grids) and fixed splits
%   for the accuracy/time trade-off.
%
%   Coarse-search trade-off: restricting the search to multiples of 16, 32
%   or 64 subcarriers costs proportionally less (time of the evaluated
%   candidates only) and loses some score; the score of the coarse optimum
%   is read from the stored metric table.

nTime = min(cfg.eval.numTimingSamples, numel(data.idxTest));
ids   = data.idxTest(1:nTime);
cand  = ds.candidates;
A     = numel(cand);

tChan = zeros(nTime, 1);  tCand = zeros(nTime, A);
tFeat = zeros(nTime, 1);  tDnn  = zeros(nTime, 1);
labelReproduced = true(nTime, 1);

% warm-up (JIT, first-call overheads)
smp = regenerateSample(cfg, ds.meta.sampleId(ids(1)));
evaluateAllocation(cfg, smp.H, smp.noiseVar, smp.ch, smp.scene, cand(1));
predictAllocation(model, smp.x);

fprintf('Timing exhaustive search vs DNN on %d test samples...\n', nTime);
for i = 1:nTime
    sid = ds.meta.sampleId(ids(i));
    smp = regenerateSample(cfg, sid);

    % ---------------- exhaustive search ------------------------------------
    t0 = tic;  H = computeTFChannel(cfg, smp.ch);  tChan(i) = toc(t0);
    SE = zeros(1, A); BER = zeros(1, A); Pd = zeros(1, A);
    for a = 1:A
        t0 = tic;
        m = evaluateAllocation(cfg, H, smp.noiseVar, smp.ch, smp.scene, cand(a));
        tCand(i, a) = toc(t0);
        SE(a) = m.SE;  BER(a) = m.BER;  Pd(a) = m.Pd;
    end
    sc = computeScore(SE, BER, Pd, smp.snr_dB, cfg.obj, (cfg.sys.N_total - cand) / cfg.sys.N_total);
    [~, b] = max(sc);
    labelReproduced(i) = (b == data.labelIdx(ids(i)));

    % ---------------- DNN --------------------------------------------------
    t0 = tic;  x = extractFeatures(cfg, smp.obs, smp.snr_dB);  tFeat(i) = toc(t0);
    [~, ~, ~, tDnn(i)] = predictAllocation(model, x);
end
if ~all(labelReproduced)
    warning('compareWithBaseline: %d re-simulated labels differ from the dataset (check seeds/config).', ...
            nnz(~labelReproduced));
end

tExh = tChan + sum(tCand, 2);
timing.exhaustive_ms      = 1e3 * mean(tExh);
timing.exhaustiveMedian_ms = 1e3 * median(tExh);
timing.channelBuild_ms    = 1e3 * mean(tChan);
timing.perCandidate_ms    = 1e3 * mean(tCand, 1);
timing.dnnFeatures_ms     = 1e3 * mean(tFeat);
timing.dnnInference_ms    = 1e3 * mean(tDnn);
timing.dnnTotal_ms        = timing.dnnFeatures_ms + timing.dnnInference_ms;
timing.dnnBatchPerSample_ms = 1e3 * results.dnnBatchTime_s / results.numTest;
timing.speedup            = timing.exhaustive_ms / timing.dnnTotal_ms;
timing.numSamples         = nTime;
timing.labelsReproduced_pct = 100 * mean(labelReproduced);

% ---------------- accuracy / time trade-off ------------------------------------
optScore = mean(results.optimal.score);
steps = cfg.alloc.step * [1 2 4 8];
steps = steps(mod(cfg.sys.N_total, steps) == 0);
idxT  = data.idxTest;
tr = struct('name', {}, 'time_ms', {}, 'pctOfOpt', {});
for s = steps
    sub = find(mod(cand, s) == 0);
    [~, k] = max(data.score(idxT, sub), [], 2);
    sc  = data.score(sub2ind(size(data.score), idxT(:), sub(k)));
    tr(end+1) = struct('name', sprintf('Search, step %d (%d cand.)', s, numel(sub)), ...
                       'time_ms', timing.channelBuild_ms + sum(timing.perCandidate_ms(sub)), ...
                       'pctOfOpt', 100 * mean(sc) / optScore); %#ok<AGROW>
end
tr(end+1) = struct('name', 'DNN', 'time_ms', timing.dnnTotal_ms, ...
                   'pctOfOpt', 100 * mean(results.dnn.score) / optScore);
tr(end+1) = struct('name', sprintf('Best fixed (N_{OFDM}=%d)', results.bestFixedNofdm), ...
                   'time_ms', NaN, 'pctOfOpt', 100 * mean(results.bestFixed.score) / optScore);
timing.tradeoff = tr;

fprintf('\n=================== COMPUTATION TIME (per decision) ===================\n');
fprintf('  Exhaustive search (%d candidates): %8.2f ms  (median %.2f ms)\n', A, timing.exhaustive_ms, timing.exhaustiveMedian_ms);
fprintf('      of which TF channel build    : %8.2f ms\n', timing.channelBuild_ms);
fprintf('  DNN total                        : %8.3f ms\n', timing.dnnTotal_ms);
fprintf('      feature extraction           : %8.3f ms\n', timing.dnnFeatures_ms);
fprintf('      inference (single sample)    : %8.3f ms\n', timing.dnnInference_ms);
fprintf('      inference (batched, /sample) : %8.4f ms\n', timing.dnnBatchPerSample_ms);
fprintf('  Speed-up (exhaustive / DNN)      : %8.1f x\n', timing.speedup);
fprintf('  Labels reproduced from seeds     : %8.1f %%\n', timing.labelsReproduced_pct);
fprintf('  Accuracy / time trade-off:\n');
for i = 1:numel(tr)
    fprintf('    %-34s %9.3f ms   %6.2f %% of optimal score\n', tr(i).name, tr(i).time_ms, tr(i).pctOfOpt);
end
fprintf('=========================================================================\n');
end
