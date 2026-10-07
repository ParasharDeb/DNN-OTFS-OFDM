function [score, parts] = computeScore(SE, BER, Pd, snr_dB, obj, otfsFraction)
%COMPUTESCORE Weighted ISAC objective for one or more candidate allocations.
%
%   [score, parts] = computeScore(SE, BER, Pd, snr_dB, obj, otfsFraction)
%
%   SE, BER, Pd  : arrays of equal size, one column per candidate allocation
%                  ([numSamples x numCandidates] or [1 x numCandidates])
%   snr_dB       : scalar or column vector (one per sample / row)
%   obj          : cfg.obj with fields alpha, beta, gamma, delta, berFloor
%   otfsFraction : [1 x numCandidates] N_OTFS / N_total of each candidate
%
%       score = alpha*SE_norm + beta*Pd - gamma*BER_norm - delta*C_norm
%
%   Normalisations (FIXED per sample, independent of which candidates are
%   compared, so the ranking is never distorted by min-max rescaling):
%     SE_norm  = SE / log2(1 + SNR)    fraction of the AWGN capacity at the
%                                      nominal SNR
%     Pd       = detection probability, already in [0, 1]
%     BER_norm = (log10 BER - log10 berFloor) / (log10 0.5 - log10 berFloor),
%                clipped to [0, 1].  Log scale: BER matters in orders of
%                magnitude; BER < berFloor counts as error-free.
%     C_norm   = (N_OTFS/N_total)^3    relative OTFS-LMMSE detection cost
%                (O(N*M^3) matrix inversions); delta = 0 by default, i.e.
%                receiver complexity is NOT part of the default objective.

assert(isequal(size(SE), size(BER), size(Pd)), 'computeScore: SE, BER, Pd sizes differ.');
assert(size(otfsFraction, 2) == size(SE, 2), 'computeScore: otfsFraction must have one entry per candidate.');

seRef   = log2(1 + 10 .^ (snr_dB(:) / 10));
seNorm  = SE ./ seRef;
logBer  = log10(max(BER, obj.berFloor));
berNorm = (logBer - log10(obj.berFloor)) / (log10(0.5) - log10(obj.berFloor));
berNorm = min(max(berNorm, 0), 1);
cNorm   = otfsFraction(:).' .^ 3;

score = obj.alpha * seNorm + obj.beta * Pd - obj.gamma * berNorm - obj.delta * cNorm;

parts.seNorm  = seNorm;
parts.berNorm = berNorm;
parts.Pd      = Pd;
parts.cNorm   = cNorm;
end
