function [x, names] = extractFeatures(cfg, obs, snr_dB)
%EXTRACTFEATURES Compact, physically meaningful DNN input vector from the observations.
%
%   [x, names] = extractFeatures(cfg, obs, snr_dB)      x : 1 x F double
%
%   Why compact statistics instead of the raw K x N complex CSI tensor
%   (2*128*32 = 8192 real numbers)?
%     * The optimal split depends on channel STATISTICS (SNR, frequency
%       selectivity, Doppler/ICI level, static-vs-dynamic power, delay
%       spread, sensing scene), not on the exact phase of each coefficient.
%       The raw phases are random nuisance variables that a small MLP would
%       have to learn to ignore, which needs far more data.
%     * Every feature below is an estimable second-order statistic of Hest
%       (correlations, power spectra) or a radar measurement.  None uses
%       hidden simulator parameters (true fD, true delay spread, profile name).
%     * 16 sub-band gains keep the coarse frequency profile, because the band
%       plan is contiguous (OFDM low, OTFS high) and the per-sub-band fading
%       matters.
%   A raw-CSI CNN is the natural next step (README, "Path to more realism").

K = cfg.sys.N_total;
N = cfg.sys.Nsym;
H = obs.Hest;
p = abs(H).^2;
pm = mean(p(:));

f = struct();
% ---- link quality --------------------------------------------------------------
f.snr_dB          = snr_dB;                                  % nominal SNR (from CQI)
f.meanGain_dB     = 10 * log10(pm);                          % large-scale/block fading
f.freqSelectivity = mean(std(p, 0, 1)) / pm;                 % fading depth across subcarriers
f.timeSelectivity = mean(std(p, 0, 2)) / pm;                 % fading change across symbols

% ---- frequency correlation (coherence bandwidth ~ 1/delay spread) ----------------
for d = cfg.csi.freqLags
    f.(sprintf('freqCorr_lag%d', d)) = abs(mean(H(1:K-d, :) .* conj(H(1+d:K, :)), 'all')) / pm;
end
% ---- time correlation (coherence time ~ 1/Doppler) -------------------------------
for d = cfg.csi.timeLags
    f.(sprintf('timeCorr_lag%d', d)) = abs(mean(H(:, 1:N-d) .* conj(H(:, 1+d:N)), 'all')) / pm;
end

% ---- power-delay profile (coarse: delay resolution 1/(K*scs)) --------------------
pdp  = mean(abs(ifft(H, [], 1)).^2, 2);
pdp  = pdp(1:K/4);                                           % causal part only
tAx  = (0:K/4-1).' / (K * cfg.sys.scs);
pp   = pdp / sum(pdp);
mu   = sum(pp .* tAx);
f.rmsDelaySpread_us = sqrt(sum(pp .* (tAx - mu).^2)) * 1e6;
f.pdpPeakRatio      = max(pdp) / sum(pdp);                   % LOS / specular indicator

% ---- Doppler power spectrum ------------------------------------------------------
S   = fftshift(mean(abs(fft(H, [], 2)).^2, 1));              % 1 x N
fAx = ((0:N-1) - floor(N/2)) / (N * cfg.sys.Tsym);
Sn  = S / sum(S);
f.staticPowerFraction = Sn(fAx == 0);                        % share of zero-Doppler (static clutter) power
f.rmsDopplerSpread_Hz = sqrt(sum(Sn .* fAx.^2));
f.iciRatio_dB         = obs.iciRatio_dB;                     % measured on null tones

% ---- coarse frequency profile -----------------------------------------------------
nb = cfg.csi.numSubbands;
pb = mean(reshape(mean(p, 2), K / nb, nb), 1);
for b = 1:nb
    f.(sprintf('subbandGain%02d_dB', b)) = 10 * log10(pb(b));
end

% ---- sensing-scene observations -----------------------------------------------------
r = obs.radar;
f.tgtSNR_dB          = r.targetSNR_dB;
f.tgtDoppler_Hz      = r.targetDoppler;
f.staticCNR_dB       = r.staticCNR_dB;
f.numMovers          = r.numMovers;
f.moverSNR_dB        = r.moverSNR_dB;
f.moverTotal_dB      = r.moverTotal_dB;
f.moverDelaySep_us   = r.moverDelaySep_us;
f.moverDopplerSep_Hz = r.moverDopplerSep;

names = fieldnames(f).';
x     = cellfun(@(n) double(f.(n)), names);

assert(all(isfinite(x)), 'extractFeatures: non-finite feature(s): %s', strjoin(names(~isfinite(x)), ', '));
end
