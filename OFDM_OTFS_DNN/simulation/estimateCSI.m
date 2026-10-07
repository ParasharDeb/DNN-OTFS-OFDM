function obs = estimateCSI(cfg, H, noiseVar, scene)
%ESTIMATECSI Imperfect channel/scene OBSERVATIONS that the DNN is allowed to see.
%
%   obs = estimateCSI(cfg, H, noiseVar, scene)
%
%   Distinguish clearly:
%     * true channel H (computeTFChannel)  -> used ONLY by the exhaustive search
%       (genie labels) and by performance evaluation;
%     * obs (this function)                -> noisy measurements a real
%       transmitter could obtain from a full-band sounding/reference frame
%       and from the previous radar frame.  The DNN features are computed
%       from obs only.
%
%   Communication CSI (complex, K x N):
%       Hest(k,n) = H(k,k,n) + e(k,n),   e ~ CN(0, noiseVar / 10^(boost/10))
%     LS estimate on a full-band sounding frame with pilot power boost
%     cfg.csi.pilotBoost_dB.  Representation: complex TF channel coefficients,
%     H = Re{H} + j*Im{H}; the DNN receives compact statistics of it (see
%     extractFeatures.m for why).
%
%   ICI measurement: on null subcarriers (every cfg.csi.nullSpacing-th tone,
%   all symbols, random QPSK on all other tones) the received power is pure
%   ICI + noise.  Subtracting the known noise power gives the ICI power.
%
%   Radar observations (previous frame, full band, i.e. finest resolution):
%   target SNR, |Doppler|, static CNR and the strongest moving clutter, each
%   with the measurement-error model in cfg.csi.radar*.

K = cfg.sys.N_total;
N = cfg.sys.Nsym;

% ---------------- communication CSI -------------------------------------------
Hr    = reshape(H, K * K, N);
dIdx  = (0:K-1) * K + (1:K);
Htrue = Hr(dIdx, :);                                         % K x N
estVar = noiseVar / 10 ^ (cfg.csi.pilotBoost_dB / 10);
obs.Hest = Htrue + sqrt(estVar / 2) * (randn(K, N) + 1j * randn(K, N));
obs.estErrVar = estVar;

% ---------------- ICI measurement on null tones --------------------------------
nullIdx = 1:cfg.csi.nullSpacing:K;
iciPow  = zeros(numel(nullIdx), N);
for n = 1:N
    x = (sign(randn(K, 1)) + 1j * sign(randn(K, 1))) / sqrt(2);   % random QPSK
    x(nullIdx) = 0;
    w = sqrt(noiseVar / 2) * (randn(numel(nullIdx), 1) + 1j * randn(numel(nullIdx), 1));
    y = H(nullIdx, :, n) * x + w;
    iciPow(:, n) = abs(y).^2;
end
iciEst = max(mean(iciPow(:)) - noiseVar, 1e-6 * noiseVar);
obs.iciRatio_dB = 10 * log10(iciEst / mean(abs(obs.Hest(:)).^2));

% ---------------- radar-side observations --------------------------------------
t   = scene.target;
mv  = scene.movers;
eDb = cfg.csi.radarPowerErr_dB;
dRes = 1 / (N * cfg.sys.Tsym);                               % Doppler resolution [Hz]
tRes = 1 / (K * cfg.sys.scs);                                % delay resolution [s]

obs.radar.targetSNR_dB  = 10 * log10(t.snr) + eDb * randn;
obs.radar.targetDoppler = abs(t.nu + dRes / sqrt(12) * randn);        % |Hz|
obs.radar.staticCNR_dB  = scene.staticCNR_dB + eDb * randn;

% Movers are only observable if detectable in the full-band map.
gFull = K * N;
detThr = 10 ^ (cfg.csi.moverDetectThreshold_dB / 10);
vis   = mv.snr * gFull > detThr;
obs.radar.numMovers = nnz(vis);
if obs.radar.numMovers > 0
    mvSnr = mv.snr(vis);  mvTau = mv.tau(vis);  mvNu = mv.nu(vis);
    [~, i] = max(mvSnr);
    obs.radar.moverSNR_dB   = 10 * log10(mvSnr(i)) + eDb * randn;
    obs.radar.moverTotal_dB = 10 * log10(sum(mvSnr)) + eDb * randn;
    obs.radar.moverDelaySep_us = abs(mvTau(i) - t.tau + tRes / sqrt(12) * randn) * 1e6;
    obs.radar.moverDopplerSep  = abs(mvNu(i) - t.nu + dRes / sqrt(12) * randn);
else
    % "No mover" encoded as a very weak mover far away in delay and Doppler.
    obs.radar.moverSNR_dB      = -60;
    obs.radar.moverTotal_dB    = -60;
    obs.radar.moverDelaySep_us = cfg.sys.Tcp * 1e6;
    obs.radar.moverDopplerSep  = 2 * cfg.sens.moverSpeedMax * cfg.sys.fc / cfg.const.c;
end
end
