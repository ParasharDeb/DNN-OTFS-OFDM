function cfg = simulationConfig()
%SIMULATIONCONFIG Every tunable parameter of the project, in one place.
%
%   cfg = simulationConfig()
%
%   Edit values here only.  Derived quantities are computed at the bottom
%   and must not be edited.  See README.md for the reasoning behind each
%   default.

%% ---------------------------------------------------------------- general
cfg.seed = 2026;                       % master seed; sample i uses seed + i

%% ---------------------------------------------------------------- numerology
cfg.sys.N_total  = 128;                % total subcarriers shared by OFDM + OTFS
cfg.sys.scs      = 15e3;               % subcarrier spacing [Hz] (LTE / NR mu=0)
cfg.sys.Nfft     = 128;                % FFT size (= N_total: no guard band, simplification)
cfg.sys.Ncp      = 16;                 % CP length [samples] -> 8.33 us (12.5 %)
cfg.sys.Nsym     = 32;                 % OFDM symbols per frame = OTFS Doppler bins
cfg.sys.fc       = 5.9e9;              % carrier [Hz] (ITS / V2X band) - for radar Doppler
cfg.sys.qamOrder = 16;                 % QAM order for BER (4, 16, 64, 256)

%% ---------------------------------------------------------------- allocation
cfg.alloc.step = 8;                    % granularity of N_OFDM (must divide N_total)

%% ---------------------------------------------------------------- comm channel
cfg.chan.profiles           = {'TDL-A', 'TDL-C', 'TDL-D'};  % 3GPP TR 38.901 (TDL-D = LOS)
cfg.chan.useToolboxProfiles = true;    % use nrTDLChannel (5G Toolbox) if installed
cfg.chan.delaySpreadRange   = [30e-9 800e-9];   % RMS delay spread [s], log-uniform
cfg.chan.maxDopplerRange    = [0 2000];         % max Doppler of moving paths [Hz]
                                                % (2 kHz @ 5.9 GHz ~ 366 km/h)
cfg.chan.dynamicFractionSet = [0 0.25 0.5 0.75 1];  % share of paths that move
cfg.chan.snrSet_dB          = 0:5:30;  % average SNR per resource element [dB]

%% ---------------------------------------------------------------- pilots
cfg.pilot.nyquistOversampling = 2;     % pilot lattice density vs. 2-D Nyquist rate

%% ---------------------------------------------------------------- CSI observations (DNN input)
cfg.csi.pilotBoost_dB           = 3;   % sounding-pilot power boost
cfg.csi.nullSpacing             = 8;   % null tones for ICI measurement
cfg.csi.numSubbands             = 16;  % sub-band gains in the feature vector
cfg.csi.freqLags                = [1 4 16];   % frequency-correlation lags [subcarriers]
cfg.csi.timeLags                = [1 4 16];   % time-correlation lags [symbols]
cfg.csi.radarPowerErr_dB        = 1;   % std of radar power measurements [dB]
cfg.csi.moverDetectThreshold_dB = 13;  % full-band SNR needed to see a mover

%% ---------------------------------------------------------------- sensing (monostatic radar)
cfg.sens.Pt_W                = 10;     % total transmit power [W] (40 dBm small cell)
cfg.sens.antennaGain_dBi     = 15;     % Tx = Rx array gain [dBi]
cfg.sens.noiseFigure_dB      = 7;
cfg.sens.systemLoss_dB       = 3;
cfg.sens.T0                  = 290;    % reference temperature [K]
cfg.sens.Pfa                 = 1e-6;   % false-alarm probability
cfg.sens.targetRCSRange_dBsm = [-20 0];% UAV / pedestrian / cyclist
cfg.sens.targetRangeMin      = 50;     % [m]  (max set by the CP, see derived)
cfg.sens.targetSpeedMax      = 40;     % max |radial speed| of target [m/s]
cfg.sens.numMoversMax        = 4;      % moving clutter objects (vehicles)
cfg.sens.moverRCSRange_dBsm  = [0 10];
cfg.sens.moverSpeedMax       = 40;     % [m/s]
cfg.sens.staticCNRRange_dB   = [20 50];% static clutter-to-noise ratio per RE
cfg.sens.clutterCancellation_dB = 40;  % zero-Doppler clutter suppression

%% ---------------------------------------------------------------- objective
% score = alpha*SE/log2(1+SNR) + beta*Pd - gamma*BER_norm   (computeScore.m)
cfg.obj.alpha    = 1.0;                % communication rate
cfg.obj.beta     = 0.5;                % sensing (detection probability)
cfg.obj.gamma    = 0.5;                % reliability (log-scaled uncoded BER)
cfg.obj.delta    = 0;                  % optional OTFS receiver-complexity penalty (off)
cfg.obj.berFloor = 1e-6;               % BER treated as error-free

%% ---------------------------------------------------------------- dataset
cfg.data.numSamples  = 12000;          % channel realisations (= samples)
cfg.data.split       = [0.70 0.15 0.15];
cfg.data.useParallel = true;           % parfor if Parallel Computing Toolbox exists
cfg.data.saveCSV     = true;

%% ---------------------------------------------------------------- DNN
cfg.dnn.mode            = 'classification';   % 'classification' | 'regression'
cfg.dnn.hiddenUnits     = [128 64 32];
cfg.dnn.dropout         = 0.1;
cfg.dnn.maxEpochs       = 150;
cfg.dnn.miniBatchSize   = 256;
cfg.dnn.initialLearnRate = 1e-3;
cfg.dnn.learnRateDropPeriod = 40;      % epochs
cfg.dnn.learnRateDropFactor = 0.5;
cfg.dnn.l2Regularization = 1e-4;
cfg.dnn.validationPatience = 20;       % early stopping (validation checks)
cfg.dnn.validationFrequencyEpochs = 1;
cfg.dnn.showTrainingPlot = false;      % MATLAB's live training window

%% ---------------------------------------------------------------- evaluation
cfg.eval.numTimingSamples = 200;       % samples re-run to time exhaustive search
cfg.eval.demoSampleIndex  = 1;         % test sample printed in the final demo

%% ---------------------------------------------------------------- link-abstraction validation
cfg.validation.enable        = true;
cfg.validation.snr_dB        = 0:5:30;
cfg.validation.dopplers      = [100 2000];
cfg.validation.nOFDM         = 64;
cfg.validation.numFrames     = 20;
cfg.validation.integerDelays = true;

%% ---------------------------------------------------------------- run control
cfg.run.regenerateDataset = false;     % false: reuse dataset/data/dataset.mat if compatible
cfg.run.retrain           = true;

%% ---------------------------------------------------------------- constants
cfg.const.c  = 299792458;              % speed of light [m/s]
cfg.const.kB = 1.380649e-23;           % Boltzmann constant [J/K]

%% ================================================================ derived (do not edit)
root = fileparts(fileparts(mfilename('fullpath')));
cfg.paths.root   = root;
cfg.paths.data   = fullfile(root, 'dataset', 'data');
cfg.paths.plots  = fullfile(root, 'plots');
cfg.paths.models = fullfile(root, 'models');

cfg.sys.Ts   = 1 / (cfg.sys.Nfft * cfg.sys.scs);              % sample period [s]
cfg.sys.Tcp  = cfg.sys.Ncp * cfg.sys.Ts;                       % CP duration [s]
cfg.sys.Tsym = (cfg.sys.Nfft + cfg.sys.Ncp) * cfg.sys.Ts;      % symbol incl. CP [s]
cfg.sys.Tframe = cfg.sys.Nsym * cfg.sys.Tsym;

cfg.alloc.candidates = 0:cfg.alloc.step:cfg.sys.N_total;      % candidate N_OFDM values
cfg.alloc.numClasses = numel(cfg.alloc.candidates);

validateConfig(cfg);
end

% ==============================================================================
function validateConfig(cfg)
assert(mod(cfg.sys.N_total, cfg.alloc.step) == 0, 'alloc.step must divide N_total.');
assert(cfg.sys.Nfft == cfg.sys.N_total, 'This prototype requires Nfft == N_total.');
assert(mod(cfg.sys.N_total, cfg.csi.numSubbands) == 0, 'numSubbands must divide N_total.');
assert(mod(cfg.sys.Nsym, 2) == 0 && mod(cfg.sys.N_total, 4) == 0, 'Nsym must be even, N_total a multiple of 4.');
assert(all(cfg.csi.freqLags < cfg.sys.N_total) && all(cfg.csi.timeLags < cfg.sys.Nsym), 'Correlation lags too large.');
assert(abs(sum(cfg.data.split) - 1) < 1e-9 && all(cfg.data.split > 0), 'data.split must be positive and sum to 1.');
assert(any(strcmp(cfg.dnn.mode, {'classification', 'regression'})), 'dnn.mode must be classification or regression.');
assert(cfg.sens.targetRangeMin < 0.9 * cfg.const.c * cfg.sys.Tcp / 2, 'targetRangeMin beyond CP-limited range.');
assert(ismember(cfg.sys.qamOrder, [4 16 64 256]), 'qamOrder must be 4, 16, 64 or 256.');
end
