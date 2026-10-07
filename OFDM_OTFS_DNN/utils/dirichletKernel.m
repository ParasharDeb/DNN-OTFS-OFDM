function D = dirichletKernel(x, L)
%DIRICHLETKERNEL Normalised periodic sinc  D(x) = (1/L) * sum_{t=0}^{L-1} exp(j*2*pi*x*t/L).
%
%   This kernel appears everywhere in multicarrier / delay-Doppler analysis:
%     * An L-point DFT of a complex exponential whose frequency is x bins
%       away from a DFT bin produces D(x) in that bin.  With x = j-k+nu/scs
%       it gives the inter-carrier interference (ICI) caused by Doppler nu.
%     * With x = (tau1-tau2)*M*scs it is the delay-domain response of an
%       M-subcarrier band (delay resolution 1/(M*scs)); with
%       x = (nu1-nu2)*N*Tsym it is the Doppler-domain response of N symbols.
%
%   D(0) = 1 and |D(x)| <= 1.  Output has the same size as x.

arguments
    x {mustBeNumeric, mustBeReal}
    L (1,1) {mustBeInteger, mustBePositive}
end

den = L * sin(pi * x / L);
D   = exp(1j * pi * x * (L - 1) / L) .* sin(pi * x) ./ den;

% At x = m*L (m integer) the ratio is 0/0; its limit is exactly 1.
singular    = abs(den) < 1e-12;
D(singular) = 1;
end
