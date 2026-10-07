function tx = generateOTFS(cfg, nOFDM)
%GENERATEOTFS Random QAM data on a delay-Doppler grid, mapped to the OTFS sub-band via ISFFT.
%
%   tx = generateOTFS(cfg, nOFDM)
%
%   OTFS on an OFDM modulator ("OFDM-based OTFS"):
%     1. M x N QAM symbols x_dd(l, k) on the delay-Doppler (DD) grid,
%        M = K - nOFDM delay bins (resolution 1/(M*scs)),
%        N Doppler bins (resolution 1/(N*Tsym)).
%     2. ISFFT (unitary) to the TF grid of the OTFS sub-band:
%          delay   -> frequency : M-point IDFT  (F_M')
%          Doppler -> time      : N-point DFT
%        so every DD symbol is spread over ALL M*N REs of the sub-band.
%        This spreading is what gives OTFS its time-frequency diversity
%        and robustness to Doppler.
%     3. The shared CP-OFDM modulator (ofdmModulate) turns the grid into a
%        time signal.  The delay-domain IDFT matches the F' used in
%        evaluateAllocation, so per-delay-bin analytical MSEs line up.
%
%   Output struct:
%     bits : [M*N*log2(Mq) x 1],  xdd : [M x N] DD symbols,
%     Xtf  : [K x N] TF grid (OTFS sub-band only),  idx : OTFS subcarriers

K  = cfg.sys.N_total;
N  = cfg.sys.Nsym;
Mq = cfg.sys.qamOrder;
kb = log2(Mq);
M  = K - nOFDM;

tx.idx  = nOFDM+1:K;
tx.bits = randi([0 1], M * N * kb, 1);
tx.Xtf  = zeros(K, N);
tx.xdd  = zeros(M, N);
if M > 0
    sym    = qammod(tx.bits, Mq, 'InputType', 'bit', 'UnitAveragePower', true);
    tx.xdd = reshape(sym, M, N);
    tx.Xtf(tx.idx, :) = isfft(tx.xdd);
end
end

function X = isfft(x)
%ISFFT Unitary inverse symplectic finite Fourier transform (DD -> TF).
[M, N] = size(x);
X = fft(ifft(x, [], 1) * sqrt(M), [], 2) / sqrt(N);
end
