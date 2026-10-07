function ch = generateChannel(cfg)
%GENERATECHANNEL Draw one random doubly-dispersive (delay + Doppler) communication channel.
%
%   ch = generateChannel(cfg)
%
%   Channel model (stated simplification, standard in the OTFS literature):
%       h(tau, t) = sum_p  g_p * exp(j*2*pi*nu_p*t) * delta(tau - tau_p)
%   i.e. P discrete paths, each with a complex gain g_p, a delay tau_p and ONE
%   Doppler shift nu_p that is constant over the frame (~2.4 ms).  Fading
%   comes from drawing new g_p for every realisation (block fading).
%
%   * Power-delay profile: 3GPP TR 38.901 TDL-A / TDL-C (NLOS) and TDL-D (LOS)
%     taken from 5G Toolbox (nrTDLChannel/info) and scaled to the drawn RMS
%     delay spread.  Without 5G Toolbox an exponential PDP is used instead.
%   * Static clutter: paths off fixed scatterers (buildings, ground) -> nu_p = 0.
%   * Dynamic components: paths via moving scatterers / a moving UE ->
%     nu_p = fD * cos(theta_p) (Jakes/Clarke angle model).  The fraction
%     of dynamic paths is drawn per realisation (cfg.chan.dynamicFractionSet).
%   * LOS profiles: the first tap is a specular component with a random but
%     deterministic phase (Rician fading); all other taps are Rayleigh.
%
%   Output struct fields:
%     profile, delaySpread [s], fDmax [Hz], dynFraction, isLOS, KfactorDB,
%     tau [P x 1, s], nu [P x 1, Hz], gain [P x 1 complex], isDynamic [P x 1],
%     tauMax [s], nuMax [Hz]
%
%   E{sum |gain|^2} = 1, so the nominal SNR is the average received SNR per
%   resource element; the instantaneous gain fluctuates (block fading).

profiles = cfg.chan.profiles;
ch.profile = profiles{randi(numel(profiles))};

[tauN, powLin, isLOS] = getDelayProfile(ch.profile, cfg.chan.useToolboxProfiles);

% --- delay spread (log-uniform between the configured limits) ----------------
dsRange = cfg.chan.delaySpreadRange;
ds = exp(log(dsRange(1)) + rand * (log(dsRange(2)) - log(dsRange(1))));
% All echoes must arrive inside the cyclic prefix, otherwise inter-symbol
% interference appears, which this model does not include.
dsMaxCP = 0.95 * cfg.sys.Tcp / max(tauN);
ds      = min(ds, dsMaxCP);
tau     = tauN(:) * ds;
P       = numel(tau);

% --- Doppler: static vs dynamic paths ----------------------------------------
fdRange = cfg.chan.maxDopplerRange;
fDmax   = fdRange(1) + rand * (fdRange(2) - fdRange(1));
dynSet  = cfg.chan.dynamicFractionSet;
dynFrac = dynSet(randi(numel(dynSet)));
isDyn   = rand(P, 1) < dynFrac;
nu      = zeros(P, 1);
nu(isDyn) = fDmax * cos(2 * pi * rand(nnz(isDyn), 1));

% --- complex path gains ------------------------------------------------------
g = sqrt(powLin(:) / 2) .* (randn(P, 1) + 1j * randn(P, 1));      % Rayleigh taps
if isLOS
    g(1) = sqrt(powLin(1)) * exp(1j * 2 * pi * rand);              % specular LOS tap
    KfactorDB = 10 * log10(powLin(1) / sum(powLin(2:end)));
else
    KfactorDB = -Inf;
end
g = g / sqrt(sum(powLin));                                           % E{sum|g|^2} = 1

ch.delaySpread = ds;
ch.fDmax       = fDmax;
ch.dynFraction = dynFrac;
ch.isLOS       = isLOS;
ch.KfactorDB   = KfactorDB;
ch.tau         = tau;
ch.nu          = nu;
ch.gain        = g;
ch.isDynamic   = isDyn;
ch.tauMax      = max(tau);
ch.nuMax       = max(abs(nu));

assert(ch.tauMax < cfg.sys.Tcp, 'generateChannel: maximum delay exceeds the cyclic prefix.');
end

% =============================================================================
function [tauN, powLin, isLOS] = getDelayProfile(profile, useToolbox)
%GETDELAYPROFILE Normalised (RMS delay spread = 1) tapped-delay-line profile.
isLOS = any(strcmpi(profile, {'TDL-D', 'TDL-E'}));
tauN = []; powLin = [];

if useToolbox && ~isempty(which('nrTDLChannel'))
    try
        tdl = nrTDLChannel;
        tdl.DelayProfile = profile;
        tdl.DelaySpread  = 1;               % delays returned in units of the delay spread
        prof = info(tdl);
        tauN   = prof.PathDelays(:);
        powLin = 10 .^ (prof.AveragePathGains(:) / 10);
    catch
        tauN = [];                          % fall through to the built-in profile
    end
end

if isempty(tauN)
    % Fallback: 12-tap exponential power-delay profile (decay constant = 1).
    tauN   = [0 0.2 0.45 0.7 1.0 1.3 1.7 2.1 2.6 3.2 4.0 5.0].';
    powLin = exp(-tauN);
    if isLOS
        powLin(1) = powLin(1) * 10^(10/10);  % ~10 dB specular first tap
    end
end

% Rescale so the profile's RMS delay spread is exactly 1 (the nominal TR 38.901
% tables already are; this makes the fallback consistent too).
p    = powLin / sum(powLin);
mu   = sum(p .* tauN);
rms  = sqrt(sum(p .* (tauN - mu).^2));
tauN = tauN / rms;
end
