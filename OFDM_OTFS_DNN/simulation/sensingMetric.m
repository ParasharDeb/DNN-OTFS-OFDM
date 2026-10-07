function s = sensingMetric(cfg, scene, M)
%SENSINGMETRIC Detection performance of the monostatic radar that uses the OTFS sub-band.
%
%   s = sensingMetric(cfg, scene, M)      M = number of OTFS subcarriers (0..K)
%
%   ASSUMPTIONS (all explicit, see README "Sensing metric"):
%   A1  Sensing is performed on the OTFS sub-band only; the OFDM sub-band is
%       used for communication.  (If the radar processed the whole band, the
%       sensing metric would not depend on the split and would drop out of
%       the optimisation.)
%   A2  The radar knows the transmitted symbols.  It divides the received TF
%       grid by them (reciprocal filtering) and applies a 2-D DFT to obtain the
%       delay-Doppler map, which is the natural OTFS domain.  A point
%       scatterer then produces a 2-D Dirichlet response with delay resolution
%       1/(M*scs) and Doppler resolution 1/(N*Tsym).  Constant-modulus
%       symbols are assumed (no noise enhancement), and the self-ICI of the
%       echo is neglected.
%   A3  Static clutter is removed by slow-time mean subtraction (zero-Doppler
%       notch).  That notch also attenuates the target by
%           g(nu) = 1 - |D_N(nu*N*Tsym)|^2      (blind-speed effect),
%       and the imperfectly cancelled residual is modelled as white noise.
%   A4  Dynamic clutter (moving vehicles) leaks into the target cell through
%       the delay and Doppler sidelobes of the rectangular-window 2-D DFT.
%   A5  Target: Swerling-I (Rayleigh-fluctuating RCS), square-law detector,
%       fixed false-alarm probability Pfa, one coherent processing interval
%       (one frame).  Then  Pd = Pfa^(1/(1+SINR)).
%
%   Post-processing SINR in the target's delay-Doppler cell:
%                 M*N * SNR_t * g(nu_t)
%   SINR = -------------------------------------------------------------
%          1 + CNR_res + M*N * sum_i SNR_i g(nu_i) |D_M(dl_i)|^2 |D_N(dk_i)|^2
%   where dl_i = (tau_i - tau_t)*M*scs and dk_i = (nu_i - nu_t)*N*Tsym are the
%   distances between mover i and the target in delay/Doppler bins.
%   Larger M gives (i) more coherent integration gain and (ii) finer range
%   resolution, i.e. better separation from moving clutter.
%
%   Output: s.Pd (0..1), s.sinr_dB, s.rangeResolution_m, s.processingGain_dB

N    = cfg.sys.Nsym;
Tsym = cfg.sys.Tsym;
scs  = cfg.sys.scs;

if M <= 0
    s.Pd = 0;  s.sinr_dB = -Inf;  s.rangeResolution_m = Inf;  s.processingGain_dB = -Inf;
    return;
end

notchGain = @(nu) 1 - abs(dirichletKernel(nu * N * Tsym, N)).^2;

t  = scene.target;
mv = scene.movers;
leak = 0;
if ~isempty(mv.snr)
    dl   = (mv.tau - t.tau) * M * scs;
    dk   = (mv.nu  - t.nu)  * N * Tsym;
    leak = sum(mv.snr .* notchGain(mv.nu) .* abs(dirichletKernel(dl, M)).^2 ...
                                         .* abs(dirichletKernel(dk, N)).^2);
end

cnrRes = 10 ^ (scene.residualCNR_dB / 10);
G      = M * N;                                      % coherent processing gain
sinr   = G * t.snr * notchGain(t.nu) / (1 + cnrRes + G * leak);

s.Pd                = cfg.sens.Pfa ^ (1 / (1 + sinr));
s.sinr_dB           = 10 * log10(sinr);
s.rangeResolution_m = cfg.const.c / (2 * M * scs);
s.processingGain_dB = 10 * log10(G);
end
