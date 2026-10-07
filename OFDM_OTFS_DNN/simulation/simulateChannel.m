function r = simulateChannel(cfg, ch, s, noiseVar)
%SIMULATECHANNEL Pass a time-domain signal through the doubly-dispersive channel + AWGN.
%
%   r = simulateChannel(cfg, ch, s, noiseVar)
%
%   Waveform-level channel used to VALIDATE the analytical TF model
%   (computeTFChannel) and the SINR->BER abstraction:
%       r(t) = sum_p g_p * exp(j*2*pi*nu_p*t) * s(t - tau_p) + w(t)
%   * Fractional delays: band-limited (ideal) interpolation, applied as a
%     linear phase in the frequency domain of the zero-padded frame.  The
%     frequency axis is [0, Fs), the same convention as computeTFChannel
%     (subcarrier j sits at j*scs).
%   * Doppler: per-sample phase rotation with absolute time t = n*Ts, so the
%     Doppler phase keeps rotating across symbols as well as inside them
%     (the latter is what creates ICI).
%   * AWGN: complex Gaussian, variance noiseVar per sample (= per RE, because
%     the modulator uses a unitary IFFT).

Ts  = 1 / (cfg.sys.Nfft * cfg.sys.scs);
L   = numel(s);
pad = ceil(max(ch.tau) / Ts) + 8;
Lp  = L + pad;

S   = fft([s(:); zeros(pad, 1)]);
f   = (0:Lp-1).' / Lp / Ts;                           % [0, Fs)
t   = (0:Lp-1).' * Ts;

r = zeros(Lp, 1);
for p = 1:numel(ch.gain)
    sp = ifft(S .* exp(-1j * 2 * pi * f * ch.tau(p)));          % s(t - tau_p)
    r  = r + ch.gain(p) * exp(1j * 2 * pi * ch.nu(p) * t) .* sp;
end
r = r(1:L);
r = r + sqrt(noiseVar / 2) * (randn(L, 1) + 1j * randn(L, 1));
end
