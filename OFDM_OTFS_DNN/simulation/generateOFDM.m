function tx = generateOFDM(cfg, nOFDM)
%GENERATEOFDM Random QAM data for the OFDM sub-band, placed directly on the TF grid.
%
%   tx = generateOFDM(cfg, nOFDM)
%
%   OFDM: one QAM symbol per time-frequency resource element (RE).  The
%   receiver equalises each RE with a single complex tap, which is optimal
%   only while the channel is constant over one OFDM symbol (no Doppler).
%
%   Output struct:
%     bits  : [nOFDM*N*log2(Mq) x 1] transmitted bits
%     Xtf   : [K x N] TF grid with QAM symbols on subcarriers 1..nOFDM, 0 elsewhere
%     idx   : subcarrier indices used by OFDM
%
%   Uses qammod (Communications Toolbox): Gray mapping, unit average power.

K  = cfg.sys.N_total;
N  = cfg.sys.Nsym;
Mq = cfg.sys.qamOrder;
kb = log2(Mq);

tx.idx  = 1:nOFDM;
tx.bits = randi([0 1], nOFDM * N * kb, 1);
tx.Xtf  = zeros(K, N);
if nOFDM > 0
    sym = qammod(tx.bits, Mq, 'InputType', 'bit', 'UnitAveragePower', true);
    tx.Xtf(tx.idx, :) = reshape(sym, nOFDM, N);
end
end
