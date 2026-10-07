function res = exhaustiveSearch(cfg, ch, scene, snr_dB, H)
%EXHAUSTIVESEARCH Conventional baseline: evaluate every candidate split, keep the best.
%
%   res = exhaustiveSearch(cfg, ch, scene, snr_dB)
%   res = exhaustiveSearch(cfg, ch, scene, snr_dB, H)   % reuse a precomputed TF channel
%
%   This is the "genie" optimiser that produces the ground-truth labels.
%   It needs the full delay-Doppler channel (all path delays, Dopplers and
%   gains) and the sensing scene, which a real system would not have.  Its
%   cost is numel(candidates) full link evaluations (dominated by the OTFS
%   LMMSE, O(N*M^3) per candidate).
%
%   Output struct (row vectors, one entry per candidate in cfg.alloc.candidates):
%     candidates, SE, BER, BER_OFDM, BER_OTFS, SINR_OFDM_dB, SINR_OTFS_dB,
%     Pd, sensSINR_dB, score, bestIdx, bestNofdm, bestNotfs, elapsed_s

cand = cfg.alloc.candidates;
A    = numel(cand);
t0   = tic;

if nargin < 5 || isempty(H)
    H = computeTFChannel(cfg, ch);
end
noiseVar = 10 ^ (-snr_dB / 10);

fields = {'SE', 'BER', 'BER_OFDM', 'BER_OTFS', 'SINR_OFDM_dB', 'SINR_OTFS_dB', 'Pd', 'sensSINR_dB'};
for f = fields, res.(f{1}) = zeros(1, A); end

for a = 1:A
    m = evaluateAllocation(cfg, H, noiseVar, ch, scene, cand(a));
    for f = fields, res.(f{1})(a) = m.(f{1}); end
end

res.score = computeScore(res.SE, res.BER, res.Pd, snr_dB, cfg.obj, ...
                         (cfg.sys.N_total - cand) / cfg.sys.N_total);
[~, res.bestIdx] = max(res.score);           % ties -> smallest nOFDM (first index)
res.candidates = cand;
res.bestNofdm  = cand(res.bestIdx);
res.bestNotfs  = cfg.sys.N_total - res.bestNofdm;
res.elapsed_s  = toc(t0);

assert(res.bestNofdm + res.bestNotfs == cfg.sys.N_total, 'Allocation constraint violated.');
end
