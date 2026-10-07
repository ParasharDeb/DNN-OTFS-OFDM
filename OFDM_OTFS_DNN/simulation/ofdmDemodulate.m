function Y = ofdmDemodulate(cfg, r)
%OFDMDEMODULATE CP removal + unitary FFT (inverse of ofdmModulate).
%
%   Y = ofdmDemodulate(cfg, r)       r : [N*(Nfft+Ncp) x 1],  Y : [K x N]

Nfft = cfg.sys.Nfft;
Ncp  = cfg.sys.Ncp;
N    = cfg.sys.Nsym;
assert(numel(r) >= N * (Nfft + Ncp), 'ofdmDemodulate: received signal too short.');

r = reshape(r(1:N * (Nfft + Ncp)), Nfft + Ncp, N);
Y = fft(r(Ncp+1:end, :), Nfft, 1) / sqrt(Nfft);
end
