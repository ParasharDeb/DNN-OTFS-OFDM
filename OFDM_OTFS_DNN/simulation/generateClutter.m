function scene = generateClutter(cfg)
%GENERATECLUTTER Draw one monostatic sensing scene: target of interest + clutter.
%
%   scene = generateClutter(cfg)
%
%   The ISAC base station transmits the OFDM/OTFS frame and listens to its
%   own echoes (monostatic radar, full-duplex assumed).  The scene contains
%     * target of interest   : small, weakly reflecting object (UAV/pedestrian/
%                              cyclist), RCS in cfg.sens.targetRCSRange_dBsm
%     * static clutter       : ground/buildings at zero Doppler, total
%                              clutter-to-noise ratio (CNR) per resource
%                              element drawn from cfg.sens.staticCNRRange_dB
%     * dynamic clutter      : 0..numMoversMax moving vehicles (strong
%                              reflectors with non-zero Doppler) that are NOT
%                              the target and leak into its delay-Doppler cell
%
%   Echo SNR per resource element (RE) from the radar range equation:
%       SNR_RE = (Pt/K) * Tu * G^2 * lambda^2 * sigma / ((4*pi)^3 * R^4 * k*T0*F*L)
%   with Tu = 1/scs the useful symbol length.  This is the same quantity as
%   radareqsnr(lambda, R, Pt/K, Tu, 'Gain', G_dB, 'RCS', sigma, ...
%              'Ts', T0*F, 'Loss', L_dB) from Phased Array System Toolbox.
%
%   Monostatic geometry:  delay = 2R/c,  Doppler = 2*v_r*fc/c.
%   Echo delays are limited to 90 % of the cyclic prefix so that CP-based
%   (OFDM-radar style) processing stays free of inter-symbol interference.
%
%   Output fields:
%     target.{range, velocity, rcs_dBsm, tau, nu, snr}   (snr linear, per RE)
%     movers.{range, velocity, rcs_dBsm, tau, nu, snr}   (column vectors)
%     staticCNR_dB, residualCNR_dB  (after zero-Doppler clutter cancellation)

c      = cfg.const.c;
fc     = cfg.sys.fc;
Rmin   = cfg.sens.targetRangeMin;
Rmax   = 0.9 * c * cfg.sys.Tcp / 2;

% --- target of interest ------------------------------------------------------
t.range    = Rmin + rand * (Rmax - Rmin);
t.velocity = (2 * rand - 1) * cfg.sens.targetSpeedMax;          % radial, m/s
t.rcs_dBsm = uniformIn(cfg.sens.targetRCSRange_dBsm);
t.tau      = 2 * t.range / c;
t.nu       = 2 * t.velocity * fc / c;
t.snr      = radarEchoSNR(cfg, t.rcs_dBsm, t.range);

% --- dynamic clutter (moving vehicles) ---------------------------------------
nMov       = randi([0, cfg.sens.numMoversMax]);
m.range    = Rmin + rand(nMov, 1) * (Rmax - Rmin);
m.velocity = (2 * rand(nMov, 1) - 1) * cfg.sens.moverSpeedMax;
m.rcs_dBsm = cfg.sens.moverRCSRange_dBsm(1) + rand(nMov, 1) * diff(cfg.sens.moverRCSRange_dBsm);
m.tau      = 2 * m.range / c;
m.nu       = 2 * m.velocity * fc / c;
m.snr      = radarEchoSNR(cfg, m.rcs_dBsm, m.range);

% --- static clutter -----------------------------------------------------------
% Zero-Doppler clutter is removed by subtracting the slow-time mean
% (background subtraction / MTI).  Imperfect cancellation (phase noise,
% wind-blown foliage, finite dynamic range) leaves a residual that is modelled
% as white noise, reduced by cfg.sens.clutterCancellation_dB.
scene.staticCNR_dB   = uniformIn(cfg.sens.staticCNRRange_dB);
scene.residualCNR_dB = scene.staticCNR_dB - cfg.sens.clutterCancellation_dB;

scene.target = t;
scene.movers = m;
end

% =============================================================================
function snr = radarEchoSNR(cfg, rcs_dBsm, R)
%RADARECHOSNR Per-resource-element echo SNR (linear) from the radar equation.
s      = cfg.sens;
lambda = cfg.const.c / cfg.sys.fc;
kB     = cfg.const.kB;
G      = 10 ^ (s.antennaGain_dBi / 10);
F      = 10 ^ (s.noiseFigure_dB / 10);
L      = 10 ^ (s.systemLoss_dB / 10);
PtRE   = s.Pt_W / cfg.sys.N_total;         % power per subcarrier
Tu     = 1 / cfg.sys.scs;                  % coherent integration time of one RE
sigma  = 10 .^ (rcs_dBsm / 10);
snr    = PtRE * Tu * G^2 * lambda^2 .* sigma ./ ((4*pi)^3 * R.^4 * kB * s.T0 * F * L);
end

function v = uniformIn(range)
v = range(1) + rand * (range(2) - range(1));
end
