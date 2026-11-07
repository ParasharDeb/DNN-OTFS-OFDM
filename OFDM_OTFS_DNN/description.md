# DNN-based OFDM/OTFS Subcarrier Allocation for ISAC

## 1. What the project is

This is a MATLAB simulation of an **Integrated Sensing and Communication (ISAC)** base station. The station shares one band of **128 subcarriers** between two waveforms:

- **OFDM** on the lower block (subcarriers `1…N_OFDM`), used only for communication.
- **OTFS** on the upper block (`N_OFDM+1…128`), used for communication and also as the **radar** (monostatic sensing) signal.

The question it answers: **for a given channel and sensing scene, how many subcarriers should go to OFDM and how many to OTFS?**

The best split depends on the conditions:

- High Doppler causes inter-carrier interference (ICI), which hurts OFDM and favours OTFS.
- OTFS costs more pilot overhead, which favours OFDM when the channel is static.
- A larger OTFS band gives more radar integration gain and finer range resolution, which improves target detection.

The conventional way to find the best split is an **exhaustive search**. It evaluates all 17 candidate splits with full link-level models and needs perfect channel knowledge. That makes it accurate but slow and not deployable.

The project trains a **deep neural network (DNN)** to predict the optimal split directly from noisy, measurable channel observations. It then compares the DNN against the exhaustive search and against fixed splits.

## 2. System parameters (defaults in `config/simulationConfig.m`)

| Parameter | Value |
|---|---|
| Total subcarriers N_total | 128 (FFT size 128, no guard band) |
| Subcarrier spacing | 15 kHz |
| Cyclic prefix | 16 samples (8.33 µs) |
| Symbols per frame (= OTFS Doppler bins) | 32 |
| Carrier frequency | 5.9 GHz (V2X band) |
| Modulation | 16-QAM |
| Allocation granularity | 8 subcarriers, so 17 candidates: N_OFDM ∈ {0, 8, …, 128} |
| Constraint | N_OFDM + N_OTFS = 128 |
| Channel models | 3GPP TDL-A, TDL-C (NLOS), TDL-D (LOS); RMS delay spread 30–800 ns |
| Max Doppler | 0–2000 Hz (up to about 366 km/h); 0/25/50/75/100 % of paths moving |
| SNR | 0 to 30 dB in 5 dB steps |
| Radar | Pt = 10 W, 15 dBi antennas, NF 7 dB, Pfa = 10⁻⁶, target RCS −20 to 0 dBsm, up to 4 moving clutter vehicles, static clutter CNR 20–50 dB with 40 dB cancellation |
| Dataset | 12,000 independent channel realisations, split 70/15/15 (train/val/test) |

## 3. Inputs

### 3a. User or configuration inputs

All tunable inputs are in `config/simulationConfig.m`: numerology, channel ranges, radar scene, objective weights, dataset size, DNN hyperparameters, and run flags (`regenerateDataset`, `retrain`). The only command is:

```matlab
>> main
```

**Toolboxes:**

- Required: Deep Learning Toolbox.
- Recommended: Communications Toolbox, used for the waveform-level validation.
- Optional: 5G Toolbox (3GPP TDL profiles; an exponential profile is used otherwise) and Parallel Computing Toolbox (parfor dataset generation).

### 3b. DNN input: 39 features per sample

The features come only from **imperfect observations** that a real transmitter could measure: an LS channel estimate from a sounding frame with pilot noise, an ICI measurement on null tones, and noisy radar measurements from the previous frame. No hidden simulator parameters are used.

| Group | Features | Count |
|---|---|---|
| Link quality | nominal SNR, mean gain, frequency selectivity, time selectivity | 4 |
| Frequency correlation | at lags 1, 4, 16 subcarriers (coherence bandwidth) | 3 |
| Time correlation | at lags 1, 4, 16 symbols (coherence time) | 3 |
| Power-delay profile | RMS delay spread, PDP peak ratio (LOS indicator) | 2 |
| Doppler spectrum | static (zero-Doppler) power fraction, RMS Doppler spread, measured ICI ratio | 3 |
| Frequency profile | 16 sub-band gains (dB) | 16 |
| Sensing scene | target SNR, target Doppler, static CNR, number of movers, strongest mover SNR, total mover power, mover delay separation, mover Doppler separation | 8 |

Features are z-score normalised using training-set statistics only.

## 4. Outputs

### 4a. DNN output

- **Classification mode (default):** a softmax over the 17 candidate splits. The prediction is the most likely class, so it is always a valid allocation `N_OFDM ∈ {0, 8, …, 128}`, with `N_OTFS = 128 − N_OFDM`.
- **Regression mode (optional):** predicts N_OFDM/128, which is then rounded and clipped to the nearest valid split.

### 4b. Ground-truth label (exhaustive search)

For each candidate split, the search computes the following with the true channel:

- **Spectral efficiency (SE):** includes CP and pilot overheads. OFDM uses a one-tap MMSE receiver with ICI treated as interference. OTFS uses joint LMMSE detection.
- **BER:** analytical 16-QAM BER from the post-equaliser SINR.
- **Detection probability Pd:** Swerling-I target, delay-Doppler radar map on the OTFS band, with clutter notch and leakage from moving clutter.

These are combined into a score, and the label is the split with the highest score:

```
score = α·SE/log2(1+SNR) + β·Pd − γ·BER_norm − δ·(N_OTFS/128)^3
defaults: α = 1.0, β = 0.5, γ = 0.5, δ = 0 (complexity penalty off)
```

The full per-candidate metric table is stored. This means labels can be recomputed for new weights without re-simulating, and the DNN's chosen split can be scored exactly.

### 4c. Files produced (all under `results/`)

- `data/dataset.mat` (+ `dataset.csv`): features, labels, and the metric table for all candidates. It is cached and reused when the config fingerprint matches.
- `models/allocationDNN.mat`: the trained network and its normalisation statistics.
- `results.mat`: per-rule test results, summary metrics, timing.
- `linkValidation.mat`: waveform-level vs analytical BER.
- `console_output.txt`: full console log of the run.
- `plots/`: 13 figures (PNG + FIG): CSI maps and ICI matrix, label distribution, training/validation curves, confusion matrix, allocation-error histogram, BER vs SNR, SE vs SNR, DNN vs optimal score and regret CDF, computation time, accuracy/time trade-off, Pd vs target SNR, allocation vs Doppler.

### 4d. Reported metrics

- **Allocation accuracy:** exact match %, within ±8 and ±16 subcarriers, MAE/RMSE in subcarriers.
- **Performance vs optimum:** score regret, % of optimal score, ΔSE %, ΔBER, ΔPd.
- **Baselines:** all-OTFS, all-OFDM, a fixed 64/64 split, and the best fixed split (chosen on the training set).
- **Speed:** exhaustive search vs DNN time per decision, speed-up factor, and comparison with coarser search grids.
- **Final demo:** one held-out channel, showing ground-truth vs predicted split, score/SE/BER/Pd for both, and inference times.

## 5. Pipeline (`main.m`)

| Stage | Action |
|---|---|
| 0 | Toolbox check, create output folders, start console log, fix random seed (2026) |
| 1 | **Link-abstraction validation:** a bit-level Monte Carlo (QAM → OFDM/OTFS → time-domain doubly-dispersive channel → equalisers → bit errors) checks that the analytical SINR→BER model behind the labels is accurate (64/64 split, Doppler 100 and 2000 Hz) |
| 2 | **Dataset generation:** for each of 12,000 samples, generate a channel, a radar scene and an SNR; compute the true time-frequency channel including ICI; produce noisy observations; extract features; run the exhaustive search over 17 splits to get the label. About 15–40 min with parfor, 1–3 h serial; cached afterwards |
| 3 | **Pre-processing:** relabel with the current weights, stratified 70/15/15 split, z-score normalisation |
| 4 | **Training:** MLP with 39 inputs → 128 → 64 → 32 units (ReLU, dropout 0.1) → 17-class softmax. Adam, learning rate 1e-3 halved every 40 epochs, L2 1e-4, up to 150 epochs, batch 256, early stopping with patience 20. About 1–3 min |
| 5 | **Testing:** score the DNN and the baselines on held-out channels by looking up the stored true metrics |
| 6 | **Timing:** exhaustive search vs DNN (feature extraction + inference) on 200 test samples |
| 7 | Generate the 13 figures |
| 8 | Save the model and results, print the final demonstration |

## 6. Current status

- The full pipeline is implemented, and the DNN trains and predicts end to end.
- All outputs are written to a single `results/` folder, and the full console output is logged to `results/console_output.txt`.
- Accuracy, regret and speed-up figures come from running `main`. See `results/console_output.txt` after a run.

## 7. Assumptions and limitations

- Block-fading channel: each path has a fixed delay and Doppler within a frame, and all echoes fall inside the CP (no inter-symbol interference).
- The labels use the analytical link model with perfect receiver CSI; pilot cost is still counted as overhead. Stage 1 validates this model.
- Sensing uses only the OTFS band. The radar is assumed to know the transmitted symbols and to run full-duplex.
- The exhaustive search uses genie information (the true channel and scene). It is a performance upper bound, not a deployable alternative.
- The DNN uses compact statistics rather than raw CSI. A CNN on raw CSI is the planned next step.

## 8. Project structure

```
OFDM_OTFS_DNN/
├── main.m                 end-to-end pipeline
├── config/                simulationConfig.m (all parameters)
├── simulation/            channel, clutter, TF channel, CSI estimation, OFDM/OTFS, BER, sensing
├── optimization/          computeScore.m, exhaustiveSearch.m
├── dataset/               generateDataset.m, extractFeatures.m, preprocessDataset.m
├── dnn/                   buildDNN.m, trainDNN.m, predictAllocation.m, testDNN.m
├── evaluation/            calculateMetrics.m, compareWithBaseline.m, plotResults.m, validateLinkAbstraction.m
├── utils/                 helpers (toolbox check, fingerprint, plotting style, sample regeneration)
└── results/               all outputs (created on first run)
```
