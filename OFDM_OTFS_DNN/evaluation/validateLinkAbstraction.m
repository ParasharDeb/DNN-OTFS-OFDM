function val = validateLinkAbstraction(cfg)
%VALIDATELINKABSTRACTION Bit-level Monte-Carlo check of the analytical BER used for the labels.
%
%   val = validateLinkAbstraction(cfg)
%
%   The exhaustive search (and hence every label) relies on an analytical
%   "link abstraction": exact post-equaliser SINR from the TF channel matrix,
%   then BER = AWGN-QAM-BER(SINR) (Gaussian approximation of ICI, leakage
%   and LMMSE residual).  This function checks that abstraction against a
%   real end-to-end waveform simulation:
%
%     bits -> QAM -> OFDM grid + OTFS (ISFFT) grid -> CP-OFDM modulator
%          -> time-domain doubly-dispersive channel + AWGN (simulateChannel)
%          -> CP removal + FFT -> OFDM: one-tap equaliser
%                                 OTFS: joint LMMSE + SFFT + bias removal
%          -> QAM demodulation -> bit errors
%
%   Sample-aligned delays (cfg.validation.integerDelays = true) are used by
%   default: with fractional delays and critical sampling (Nfft = K, no guard
%   band), band-limited interpolation of the CP frame adds about -23 dB of
%   inter-symbol leakage that the analytical model does not contain.  That
%   leakage is a property of the critical-sampling prototype, not of the
%   ICI/SINR model.  Set integerDelays = false to see its effect.
%
%   Output: val.snr_dB, val.dopplers, val.berSim{OFDM,OTFS}, val.berAna{OFDM,OTFS}
%   ([numDopplers x numSNR]), and a figure saved in cfg.paths.plots.

v      = cfg.validation;
cfgV   = cfg;
cfgV.chan.dynamicFractionSet = 1;                % every path moves: worst-case ICI
a      = v.nOFDM;
K      = cfg.sys.N_total;
N      = cfg.sys.Nsym;
M      = K - a;
Mq     = cfg.sys.qamOrder;
t      = a+1:K;
F      = fft(eye(M)) / sqrt(M);
Ts     = 1 / (cfg.sys.Nfft * cfg.sys.scs);

nD = numel(v.dopplers);  nS = numel(v.snr_dB);
val.snr_dB   = v.snr_dB;
val.dopplers = v.dopplers;
val.nOFDM    = a;
[val.berSimOFDM, val.berSimOTFS, val.berAnaOFDM, val.berAnaOTFS] = deal(zeros(nD, nS));

fprintf('  Link-abstraction validation: split %d/%d, %d frames per point\n', a, M, v.numFrames);
for d = 1:nD
    cfgV.chan.maxDopplerRange = v.dopplers(d) * [1 1];
    for si = 1:nS
        rng(cfg.seed + 7919 * d + si, 'twister');
        nv = 10 ^ (-v.snr_dB(si) / 10);
        eO = 0; bO = 0; eT = 0; bT = 0; aO = 0; aT = 0;
        for fr = 1:v.numFrames
            ch = generateChannel(cfgV);
            if v.integerDelays
                ch.tau    = round(ch.tau / Ts) * Ts;
                ch.tauMax = max(ch.tau);
            end
            H = computeTFChannel(cfgV, ch);

            % ---------- waveform-level transmission --------------------------
            txO = generateOFDM(cfgV, a);
            txT = generateOTFS(cfgV, a);
            s   = ofdmModulate(cfgV, txO.Xtf + txT.Xtf);
            r   = simulateChannel(cfgV, ch, s, nv);
            Y   = ofdmDemodulate(cfgV, r);

            % OFDM: one-tap equaliser with perfect CSI
            Hr  = reshape(H, K * K, N);
            Hd  = Hr((0:K-1) * K + (1:K), :);
            Xo  = Y(1:a, :) ./ Hd(1:a, :);
            bh  = qamdemod(Xo(:), Mq, 'OutputType', 'bit', 'UnitAveragePower', true);
            eO  = eO + sum(bh ~= txO.bits);   bO = bO + numel(txO.bits);

            % OTFS: joint LMMSE per OFDM symbol (TF), SFFT, remove LMMSE bias
            Xt  = zeros(M, N);  mse = zeros(M, 1);
            for n = 1:N
                Hn = H(t, :, n);
                Gm = Hn(:, t);
                Cn = Hn * Hn' + nv * eye(M);
                Xt(:, n) = Gm' * (Cn \ Y(t, n));
                GF  = Gm * F';
                mse = mse + (1 - real(sum(conj(GF) .* (Cn \ GF), 1))).';
            end
            mse = min(max(mse / N, 1e-12), 1);
            zdd = ifft(fft(Xt, [], 1) / sqrt(M), [], 2) * sqrt(N);   % SFFT (inverse of ISFFT)
            zdd = zdd ./ (1 - mse);                                  % unbiased LMMSE estimate
            bh  = qamdemod(zdd(:), Mq, 'OutputType', 'bit', 'UnitAveragePower', true);
            eT  = eT + sum(bh ~= txT.bits);   bT = bT + numel(txT.bits);

            % ---------- analytical prediction (same channel) ------------------
            m  = evaluateAllocation(cfgV, H, nv, ch, generateClutter(cfgV), a);
            aO = aO + m.BER_OFDM;   aT = aT + m.BER_OTFS;
        end
        val.berSimOFDM(d, si) = eO / bO;   val.berSimOTFS(d, si) = eT / bT;
        val.berAnaOFDM(d, si) = aO / v.numFrames;
        val.berAnaOTFS(d, si) = aT / v.numFrames;
        fprintf('    fD=%5.0f Hz SNR=%4.1f dB | OFDM sim %.2e ana %.2e | OTFS sim %.2e ana %.2e\n', ...
            v.dopplers(d), v.snr_dB(si), val.berSimOFDM(d, si), val.berAnaOFDM(d, si), ...
            val.berSimOTFS(d, si), val.berAnaOTFS(d, si));
    end
end

% ------------------------------- plot ---------------------------------------
fig = figure('Name', 'Link abstraction validation', 'Color', 'w', 'Position', [100 100 760 520]);
ax = axes(fig); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
cols = lines(nD);  floorB = 1e-6;  leg = {};
for d = 1:nD
    semilogy(ax, v.snr_dB, max(val.berAnaOFDM(d, :), floorB), '-',  'Color', cols(d, :), 'LineWidth', 1.6);
    semilogy(ax, v.snr_dB, max(val.berSimOFDM(d, :), floorB), 'o',  'Color', cols(d, :), 'LineWidth', 1.4, 'MarkerSize', 7);
    semilogy(ax, v.snr_dB, max(val.berAnaOTFS(d, :), floorB), '--', 'Color', cols(d, :), 'LineWidth', 1.6);
    semilogy(ax, v.snr_dB, max(val.berSimOTFS(d, :), floorB), 's',  'Color', cols(d, :), 'LineWidth', 1.4, 'MarkerSize', 7);
    leg = [leg, {sprintf('OFDM analytical, f_D=%g Hz', v.dopplers(d)), sprintf('OFDM simulated, f_D=%g Hz', v.dopplers(d)), ...
                 sprintf('OTFS analytical, f_D=%g Hz', v.dopplers(d)), sprintf('OTFS simulated, f_D=%g Hz', v.dopplers(d))}]; %#ok<AGROW>
end
set(ax, 'YScale', 'log', 'FontSize', 12);
xlabel(ax, 'SNR per resource element (dB)');  ylabel(ax, 'Uncoded bit error rate');
title(ax, sprintf('Link abstraction vs. waveform simulation (%d-QAM, split %d/%d)', Mq, a, M));
legend(ax, leg, 'Location', 'southwest', 'FontSize', 9);
ylim(ax, [floorB 1]);
saveFigure(fig, cfg, '00_link_abstraction_validation');
end
