function m = evaluateAllocation(cfg, H, noiseVar, ch, scene, nOFDM)
%EVALUATEALLOCATION Communication + sensing performance of one OFDM/OTFS split.
%
%   m = evaluateAllocation(cfg, H, noiseVar, ch, scene, nOFDM)
%
%   Inputs
%     H        : [K x K x N] TF channel incl. ICI (computeTFChannel)
%     noiseVar : noise variance per RE (= 1/SNR, unit-power symbols)
%     ch       : channel struct (for pilot design: tauMax, nuMax)
%     scene    : sensing scene (generateClutter)
%     nOFDM    : number of OFDM subcarriers (0..K, multiple of the step)
%
%   Band plan: subcarriers 1..nOFDM -> OFDM, nOFDM+1..K -> OTFS (M = K-nOFDM).
%   Every subcarrier carries a unit-power symbol, so total transmit power is
%   the same for every split.
%
%   Receivers (perfect CSI at the receiver; pilot COST is still accounted for):
%   * OFDM: one-tap MMSE equaliser per RE.  ICI from ALL other subcarriers
%     (OFDM and OTFS) is interference:
%         SINR_kn = |H(k,k,n)|^2 / (sum_{j~=k} |H(k,j,n)|^2 + sigma^2)
%   * OTFS: joint LMMSE detection of all M*N DD symbols, which equalises the
%     ICI inside the OTFS band.  Leakage from the OFDM band is treated as
%     Gaussian interference.  Because ISFFT is unitary and the TF channel is
%     block-diagonal over symbols, the MSE of the DD symbol in delay bin l is
%         MSE_l = 1 - (1/N) sum_n [ (G_n F')' * C_n^{-1} * (G_n F') ]_{ll}
%     with G_n = H(otfs,otfs,n), C_n = H(otfs,:,n)H(otfs,:,n)' + sigma^2 I and
%     F the M-point unitary DFT.  Unbiased SINR_l = 1/MSE_l - 1.
%
%   Pilot overheads (fraction of REs that cannot carry data):
%   * OFDM: rectangular pilot lattice satisfying 2-D Nyquist sampling with
%     oversampling factor 'os':  Df = floor(1/(os*tauMax*scs)) subcarriers,
%     Dt = floor(1/(2*os*nuMax*Tsym)) symbols  ->  ovO = 1/(Df*Dt).
%   * OTFS: embedded DD pilot with a guard region spanning +/- lMax delay
%     bins and the whole Doppler axis (needed for fractional Doppler):
%     lMax = ceil(tauMax*M*scs)  ->  ovT = (2*lMax+1)/M.
%
%   Outputs (struct): nOFDM, nOTFS, SE [bit/s/Hz], BER (bit-weighted over both
%   waveforms), BER_OFDM, BER_OTFS, SINR_OFDM_dB, SINR_OTFS_dB (mean of
%   the linear SINR), overheadOFDM, overheadOTFS, Pd, sensSINR_dB, rangeRes_m

K    = cfg.sys.N_total;
N    = cfg.sys.Nsym;
scs  = cfg.sys.scs;
Mq   = cfg.sys.qamOrder;
os   = cfg.pilot.nyquistOversampling;

assert(isequal(size(H), [K K N]), 'evaluateAllocation: H must be K x K x N.');
assert(nOFDM >= 0 && nOFDM <= K && nOFDM == round(nOFDM), 'evaluateAllocation: invalid nOFDM.');

M     = K - nOFDM;
tauMx = max(ch.tauMax, 1 / (K * scs));       % at least one sample of delay spread
nuMx  = max(ch.nuMax, 1e-3);

m.nOFDM = nOFDM;
m.nOTFS = M;

seSum = 0;  bitErr = 0;  nData = 0;

% ---------------------------- OFDM sub-band ----------------------------------
if nOFDM > 0
    Df  = max(1, floor(1 / (os * tauMx * scs)));
    Dt  = min(N, max(1, floor(1 / (2 * os * nuMx * cfg.sys.Tsym))));
    ovO = 1 / (Df * Dt);

    Hr    = reshape(H, K * K, N);
    dIdx  = (0:K-1) * K + (1:K);                     % linear indices of diag(H(:,:,n))
    pSig  = abs(Hr(dIdx, :)).^2;                     % K x N, desired power
    pTot  = squeeze(sum(abs(H).^2, 2));              % K x N, desired + ICI
    sinrO = pSig(1:nOFDM, :) ./ (pTot(1:nOFDM, :) - pSig(1:nOFDM, :) + noiseVar);

    berO  = mean(qamBER(sinrO(:), Mq));
    nDO   = nOFDM * N * (1 - ovO);
    seSum = seSum + (1 - ovO) * sum(log2(1 + sinrO(:)));
    bitErr = bitErr + berO * nDO;   nData = nData + nDO;

    m.BER_OFDM     = berO;
    m.SINR_OFDM_dB = 10 * log10(mean(sinrO(:)));
    m.overheadOFDM = ovO;
else
    m.BER_OFDM = NaN;  m.SINR_OFDM_dB = NaN;  m.overheadOFDM = NaN;
end

% ---------------------------- OTFS sub-band ----------------------------------
if M > 0
    lMax = ceil(tauMx * M * scs);
    ovT  = min(1, (2 * lMax + 1) / M);

    t   = nOFDM+1:K;
    F   = fft(eye(M)) / sqrt(M);                     % unitary DFT (delay <-> frequency)
    mse = zeros(M, 1);
    for n = 1:N
        Hn = H(t, :, n);                             % OTFS rows, all transmitted subcarriers
        GF = Hn(:, t) * F';                          % DD(delay) -> received TF
        Cn = Hn * Hn' + noiseVar * eye(M);           % received covariance (signal+leakage+noise)
        W  = Cn \ GF;
        mse = mse + (1 - real(sum(conj(GF) .* W, 1))).';
    end
    mse   = min(max(mse / N, 1e-12), 1);
    sinrT = 1 ./ mse - 1;

    berT  = mean(qamBER(sinrT, Mq));
    nDT   = M * N * (1 - ovT);
    seSum = seSum + (1 - ovT) * N * sum(log2(1 + sinrT));
    bitErr = bitErr + berT * nDT;   nData = nData + nDT;

    m.BER_OTFS     = berT;
    m.SINR_OTFS_dB = 10 * log10(mean(sinrT));
    m.overheadOTFS = ovT;
else
    m.BER_OTFS = NaN;  m.SINR_OTFS_dB = NaN;  m.overheadOTFS = NaN;
end

% ---------------------------- totals -----------------------------------------
cpEff = cfg.sys.Nfft / (cfg.sys.Nfft + cfg.sys.Ncp);       % CP time overhead
m.SE  = cpEff * seSum / (K * N);                            % bit/s/Hz over the whole band
if nData > 0
    m.BER = bitErr / nData;
else
    m.BER = 0.5;                                            % no data symbols at all
end

% ---------------------------- sensing ----------------------------------------
s = sensingMetric(cfg, scene, M);
m.Pd          = s.Pd;
m.sensSINR_dB = s.sinr_dB;
m.rangeRes_m  = s.rangeResolution_m;
end
