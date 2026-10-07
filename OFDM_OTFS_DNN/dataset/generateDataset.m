function ds = generateDataset(cfg)
%GENERATEDATASET Simulate channels, observations and exhaustive-search labels.
%
%   ds = generateDataset(cfg)
%
%   For every sample i (one independent channel realisation):
%     1. generateChannel   -> comm channel (delays, Dopplers, gains)
%     2. generateClutter   -> sensing scene (target, static + moving clutter)
%     3. SNR drawn from cfg.chan.snrSet_dB
%     4. computeTFChannel  -> exact TF channel incl. ICI          (TRUE CSI)
%     5. estimateCSI       -> noisy observations                  (DNN INPUT side)
%     6. extractFeatures   -> feature vector x_i
%     7. exhaustiveSearch  -> metrics of EVERY candidate split + best split
%                             (ground-truth label y_i = N_OFDM*)
%
%   The full per-candidate metric table is stored, so that
%     * the DNN's chosen split can be scored exactly at test time, and
%     * labels can be recomputed for other objective weights without
%       re-simulating (see preprocessDataset / relabel).
%
%   Saved to cfg.paths.data/dataset.mat (+ dataset.csv if cfg.data.saveCSV).

nS = cfg.data.numSamples;
A  = cfg.alloc.numClasses;
if ~exist(cfg.paths.data, 'dir'), mkdir(cfg.paths.data); end

useParfor = cfg.data.useParallel && ~isempty(ver('parallel')) && ...
            license('test', 'Distrib_Computing_Toolbox');

fprintf('Generating %d samples x %d candidate splits (%s)...\n', nS, A, ...
        ternary(useParfor, 'parfor', 'serial for-loop'));
t0 = tic;
out = cell(nS, 1);
if useParfor
    if isempty(gcp('nocreate')), parpool; end
    q = parallel.pool.DataQueue;
    afterEach(q, @(~) progress(nS, t0));
    progress(-1);                                  % reset counter
    parfor i = 1:nS
        out{i} = simulateOneSample(cfg, i);
        send(q, i);
    end
else
    progress(-1);
    for i = 1:nS
        out{i} = simulateOneSample(cfg, i);
        progress(nS, t0);
    end
end
fprintf('Dataset generated in %.1f min.\n', toc(t0) / 60);

% ------------------------------------------------------------ assemble arrays
s1 = out{1};
ds.featureNames = s1.featureNames;
ds.candidates   = cfg.alloc.candidates;
ds.N_total      = cfg.sys.N_total;
ds.X            = cell2mat(cellfun(@(s) s.x, out, 'UniformOutput', false));          % nS x F

mFields = fieldnames(s1.metrics);
for f = mFields.'
    ds.metrics.(f{1}) = cell2mat(cellfun(@(s) s.metrics.(f{1}), out, 'UniformOutput', false));  % nS x A
end
metaFields = fieldnames(s1.meta);
for f = metaFields.'
    ds.meta.(f{1}) = cell2mat(cellfun(@(s) s.meta.(f{1}), out, 'UniformOutput', false));        % nS x 1
end
ds.labelIdx  = ds.meta.bestIdx;
ds.Y_nOFDM   = ds.candidates(ds.labelIdx).';
ds.Y_nOTFS   = ds.N_total - ds.Y_nOFDM;
ds.objUsed   = cfg.obj;
ds.fingerprint = datasetFingerprint(cfg);
ds.created   = char(datetime('now'));

% ------------------------------------------------------------ validation
assert(all(isfinite(ds.X), 'all'), 'Dataset contains non-finite features.');
assert(all(ds.Y_nOFDM + ds.Y_nOTFS == cfg.sys.N_total), 'N_OFDM + N_OTFS ~= N_total.');
assert(isequal(size(ds.metrics.SE), [nS A]), 'Metric table has the wrong size.');

% ------------------------------------------------------------ save
save(fullfile(cfg.paths.data, 'dataset.mat'), 'ds', '-v7.3');
if cfg.data.saveCSV
    T = array2table(ds.X, 'VariableNames', ds.featureNames);
    T.SNR_dB      = ds.meta.snr_dB;
    T.fDmax_Hz    = ds.meta.fDmax;
    T.delaySpread_ns = ds.meta.delaySpread * 1e9;
    T.dynFraction = ds.meta.dynFraction;
    T.isLOS       = ds.meta.isLOS;
    T.Y_nOFDM     = ds.Y_nOFDM;
    T.Y_nOTFS     = ds.Y_nOTFS;
    writetable(T, fullfile(cfg.paths.data, 'dataset.csv'));
end
fprintf('Saved dataset to %s\n', cfg.paths.data);
end

% ==============================================================================
function s = simulateOneSample(cfg, i)
rng(cfg.seed + i, 'twister');                         % per-sample seed: parfor-safe

ch    = generateChannel(cfg);
scene = generateClutter(cfg);
snr   = cfg.chan.snrSet_dB(randi(numel(cfg.chan.snrSet_dB)));
nv    = 10 ^ (-snr / 10);

tH  = tic;
H   = computeTFChannel(cfg, ch);
tH  = toc(tH);

obs = estimateCSI(cfg, H, nv, scene);
[s.x, s.featureNames] = extractFeatures(cfg, obs, snr);

res = exhaustiveSearch(cfg, ch, scene, snr, H);

for f = {'SE', 'BER', 'BER_OFDM', 'BER_OTFS', 'SINR_OFDM_dB', 'SINR_OTFS_dB', 'Pd', 'sensSINR_dB', 'score'}
    s.metrics.(f{1}) = res.(f{1});
end

s.meta.sampleId     = i;
s.meta.snr_dB       = snr;
s.meta.bestIdx      = res.bestIdx;
s.meta.searchTime_s = res.elapsed_s + tH;
s.meta.profileIdx   = find(strcmp(cfg.chan.profiles, ch.profile));
s.meta.delaySpread  = ch.delaySpread;
s.meta.fDmax        = ch.fDmax;
s.meta.dynFraction  = ch.dynFraction;
s.meta.isLOS        = double(ch.isLOS);
s.meta.KfactorDB    = ch.KfactorDB;
s.meta.tauMax       = ch.tauMax;
s.meta.nuMax        = ch.nuMax;
s.meta.tgtSNR_dB    = 10 * log10(scene.target.snr);
s.meta.tgtRange     = scene.target.range;
s.meta.tgtVelocity  = scene.target.velocity;
s.meta.tgtRCS_dBsm  = scene.target.rcs_dBsm;
s.meta.numMovers    = numel(scene.movers.snr);
s.meta.staticCNR_dB = scene.staticCNR_dB;
end

function progress(nS, t0)
persistent count
if nS < 0, count = 0; return; end
count = count + 1;
if mod(count, max(1, round(nS / 20))) == 0 || count == nS
    el = toc(t0);
    fprintf('  %6d / %d samples  (%.0f s elapsed, ~%.0f s left)\n', count, nS, el, el / count * (nS - count));
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
