function fp = datasetFingerprint(cfg)
%DATASETFINGERPRINT Text signature of every setting that changes the SIMULATED data.
%
%   The objective weights (cfg.obj) are deliberately excluded: labels are
%   recomputed from the stored per-candidate metrics, so changing alpha/beta/
%   gamma never requires re-simulation.  The DNN settings are excluded too.

s.seed  = cfg.seed;
s.sys   = rmfield(cfg.sys, intersect(fieldnames(cfg.sys), {'Ts', 'Tcp', 'Tsym', 'Tframe'}));
s.alloc = cfg.alloc.step;
s.chan  = cfg.chan;
s.pilot = cfg.pilot;
s.csi   = cfg.csi;
s.sens  = cfg.sens;
s.n     = cfg.data.numSamples;
fp = jsonencode(s);
end
