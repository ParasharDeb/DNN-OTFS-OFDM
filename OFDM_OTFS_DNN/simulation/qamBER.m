function ber = qamBER(sinr, qamOrder)
%QAMBER Uncoded Gray-mapped square M-QAM bit error rate at a given (linear) SINR.
%
%   ber = qamBER(sinr, qamOrder)
%
%   The residual interference after equalisation (ICI, inter-waveform
%   leakage, LMMSE residual) is treated as Gaussian, so the AWGN BER curve
%   is evaluated at the post-equaliser SINR.  This is the standard Gaussian
%   approximation; validateLinkAbstraction.m checks it against a bit-level
%   Monte-Carlo simulation.
%
%   Uses berawgn (Communications Toolbox) to build a lookup table once and
%   then interpolates log10(BER) on a fine SINR grid in dB, which is much
%   faster than calling berawgn for every resource element.  Without
%   Communications Toolbox, the exact square-QAM expression is used.

arguments
    sinr {mustBeNumeric, mustBeReal}
    qamOrder (1,1) {mustBeMember(qamOrder, [4 16 64 256])}
end

persistent tblOrder tblSinrDb tblLogBer
if isempty(tblOrder) || tblOrder ~= qamOrder
    tblSinrDb = (-30:0.05:60).';
    tblLogBer = log10(max(berTable(tblSinrDb, qamOrder), 1e-300));
    tblOrder  = qamOrder;
end

sinrDb = 10 * log10(max(sinr, 1e-12));
sinrDb = min(max(sinrDb, tblSinrDb(1)), tblSinrDb(end));
ber    = 10 .^ interp1(tblSinrDb, tblLogBer, sinrDb, 'linear');
ber    = reshape(ber, size(sinr));
end

% =============================================================================
function ber = berTable(sinrDb, Mq)
k = log2(Mq);
if ~isempty(which('berawgn'))
    ebno = sinrDb - 10 * log10(k);            % Es/N0 -> Eb/N0
    ber  = berawgn(ebno, 'qam', Mq);
    ber  = ber(:);
else
    % Exact BER of Gray-coded square M-QAM (Cho & Yoon, IEEE Trans. Commun., 2002).
    g    = 10 .^ (sinrDb / 10);
    L    = sqrt(Mq);
    ber  = zeros(size(g));
    qf   = @(x) 0.5 * erfc(x / sqrt(2));
    for kk = 1:log2(L)
        Pk = zeros(size(g));
        for i = 0:((1 - 2^(-kk)) * L - 1)
            w  = (-1)^floor(i * 2^(kk - 1) / L) * (2^(kk - 1) - floor(i * 2^(kk - 1) / L + 0.5));
            Pk = Pk + w * 2 * qf((2 * i + 1) * sqrt(3 * g / (Mq - 1)));
        end
        ber = ber + Pk / L;
    end
    ber = ber / log2(L);
end
ber = min(max(ber, 0), 0.5);
end
