function s = ofdmModulate(cfg, Xtf)
%OFDMMODULATE Shared CP-OFDM modulator for the composite OFDM + OTFS frame.
%
%   s = ofdmModulate(cfg, Xtf)       Xtf : [K x N],  s : [N*(Nfft+Ncp) x 1]
%
%   Unitary IFFT per symbol (so the time-domain sample power equals the
%   average RE power) followed by cyclic-prefix insertion.  The CP turns the
%   linear channel convolution into a circular one as long as every path
%   delay is shorter than the CP.  Equivalent to ofdmmod (Communications
%   Toolbox) apart from subcarrier ordering and scaling, which are kept
%   explicit here so they match the analytical model in computeTFChannel.

K    = cfg.sys.N_total;
Nfft = cfg.sys.Nfft;
Ncp  = cfg.sys.Ncp;
assert(Nfft == K, 'ofdmModulate: this prototype assumes Nfft == N_total.');
assert(size(Xtf, 1) == K, 'ofdmModulate: Xtf must have K rows.');

x = ifft(Xtf, Nfft, 1) * sqrt(Nfft);           % Nfft x N
x = [x(end-Ncp+1:end, :); x];                  % add cyclic prefix
s = x(:);
end
