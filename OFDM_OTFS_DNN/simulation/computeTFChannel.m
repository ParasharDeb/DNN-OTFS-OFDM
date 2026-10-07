function H = computeTFChannel(cfg, ch)
%COMPUTETFCHANNEL Exact time-frequency (TF) channel of a CP-multicarrier frame, including ICI.
%
%   H = computeTFChannel(cfg, ch)      H : [K x K x N] complex
%
%   Both waveforms share one CP-OFDM modulator (K subcarriers, N symbols):
%     * OFDM  : QAM symbols are placed directly on the TF grid.
%     * OTFS  : QAM symbols live on a delay-Doppler (DD) grid and are mapped
%               to the TF grid with the ISFFT ("OFDM-based OTFS").
%   So a single TF channel description serves both waveforms.
%
%   After CP removal and FFT, OFDM symbol n is received as
%       Y_n = H(:,:,n) * X_n + W_n,
%   with (derivation in README, Section "Channel model"):
%       H(k,j,n) = sum_p g_p * exp(j2*pi*nu_p*t_n) * exp(-j2*pi*j*scs*tau_p)
%                        * D_K( j - k + nu_p/scs )
%       t_n = n*Tsym + Tcp  (start of the useful part of symbol n)
%   D_K is the K-point Dirichlet kernel (dirichletKernel.m).
%     * Diagonal entries   H(k,k,n): the per-subcarrier channel seen by a
%                                     one-tap equaliser.
%     * Off-diagonal entries        : inter-carrier interference caused by
%                                     Doppler (nu_p ~= 0).  It also couples the
%                                     OFDM and OTFS sub-bands to each other
%                                     (inter-waveform interference).
%   Valid while every tau_p < Tcp (no inter-symbol interference), which
%   generateChannel enforces.  Simplification: Nfft = K (no oversampling, no
%   guard band), so ICI wraps around the band edges circularly.

K   = cfg.sys.N_total;
N   = cfg.sys.Nsym;
scs = cfg.sys.scs;
P   = numel(ch.gain);

kIdx = 0:K-1;
off  = kIdx - kIdx.';                    % off(k+1, j+1) = j - k

% G(:, p): K*K TF response of path p without its Doppler phase rotation.
G = zeros(K * K, P);
for p = 1:P
    Dp = dirichletKernel(off + ch.nu(p) / scs, K);                     % ICI pattern
    Gp = Dp .* exp(-1j * 2 * pi * scs * ch.tau(p) * kIdx);            % delay -> phase ramp over j
    G(:, p) = Gp(:);
end

% Phase of every path at the start of each OFDM symbol (Doppler rotation).
tn = (0:N-1) * cfg.sys.Tsym + cfg.sys.Tcp;                             % 1 x N
C  = ch.gain(:) .* exp(1j * 2 * pi * ch.nu(:) * tn);                   % P x N

H = reshape(G * C, K, K, N);
end
