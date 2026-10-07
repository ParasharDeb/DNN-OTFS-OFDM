function smp = regenerateSample(cfg, sampleId)
%REGENERATESAMPLE Re-create dataset sample 'sampleId' bit-exactly from its seed.
%
%   smp = regenerateSample(cfg, sampleId)
%
%   Uses exactly the same random-number sequence as generateDataset/
%   simulateOneSample, so the channel, the sensing scene, the SNR and the
%   observations are identical to those stored in the dataset.  Used for
%   timing the exhaustive search, the final demo, and the CSI plots.
%
%   Output fields: ch, scene, snr_dB, noiseVar, H, obs, x (features)

rng(cfg.seed + sampleId, 'twister');
smp.ch     = generateChannel(cfg);
smp.scene  = generateClutter(cfg);
smp.snr_dB = cfg.chan.snrSet_dB(randi(numel(cfg.chan.snrSet_dB)));
smp.noiseVar = 10 ^ (-smp.snr_dB / 10);
smp.H      = computeTFChannel(cfg, smp.ch);
smp.obs    = estimateCSI(cfg, smp.H, smp.noiseVar, smp.scene);
smp.x      = extractFeatures(cfg, smp.obs, smp.snr_dB);
end
