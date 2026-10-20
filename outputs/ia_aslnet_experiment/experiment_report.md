# ASL identifiability-aware experiment report

**Run status:** PRELIMINARY SMOKE TEST.

## Question and simulator
Tests whether local information in multi-PLD ASL signals predicts ATT estimation errors and whether timing-conditioned spatial estimation changes errors. The forward equation and SCALE=100000 follow the reference implementation. The observation is a sum of two Rician magnitudes. High-SNR signal consistency uses the explicitly approximate `2 * noiseless_signal`; it is not an exact likelihood.

## Data and models
Independent synthetic brain phantoms were generated at 64x64. Training/validation/test phantom IDs are disjoint. Normalization was fitted on valid training voxels only. Baseline is the reference-style four-channel spatial U-Net with widths [8, 16, 32, 64]; IA-ASLNet uses shared per-PLD features and actual PLD conditioning. Full reference width is [32,64,128,256]. Smoke-width/model and short epochs are not baseline reproduction.

## Training and evaluation
Device: cuda; train/validation/test counts: 12/4/6. Metrics are computed per phantom inside the valid tissue mask and then aggregated across phantoms. No clipping is used. Locked stress conditions were generated after training and not used for selection.

## Results
           att_rmse   att_mae  att_bias  att_pearson_r   att_ccc   cbf_rmse    cbf_mae  cbf_bias  cbf_pearson_r   cbf_ccc  inference_seconds_per_phantom  parameter_count
model                                                                                                                                                                    
baseline   0.376112  0.306735  0.014294       0.550029  0.027941  13.515948  11.951654  2.588755       0.787457  0.027986                       0.005519         486634.0
ia_timing  0.371269  0.313113 -0.038666       0.633945  0.059059  13.594317  10.016073 -3.942466       0.770002  0.055230                       0.006563         488492.0

IA minus baseline mean ATT RMSE (seconds): -0.004843. IA timing model had lower mean ATT RMSE on this run.
Paired phantom bootstrap intervals are saved in `paired_phantom_bootstrap_ci.csv`; this resamples whole phantoms, not pixels.
The paired phantom bootstrap interval includes zero; this run does not establish an ATT improvement.
The highest per-phantom ATT RMSE occurred for `test-7-1` under `ia_timing`: 0.4519 s.

## Ablations
Ablations were not enabled for this run; no ablation result is claimed.

## Identifiability
The local information analysis scales CBF and ATT by their full configured ranges and uses a high-SNR Gaussian variance approximation (`2 sigma^2`) to the sum-of-Rician noise. Error association is computed at phantom level and is descriptive at smoke sample sizes. See `identifiability_timing_analysis.csv` and `test_phantom_identifiability.csv`.
Phantom-level Spearman rho=0.3143, p=0.5441, n=6; too few phantoms for the predeclared nuisance-adjusted model.

## Stress tests and caveats
See `stress_test_per_phantom.csv`. Synthetic generator assumptions, 2-D morphology, known noise calibration, limited sample size, and no real ASL-MRI validation limit any conclusion. The smoke run is a pipeline check, not a final scientific claim. The next justified experiment is a larger matched multi-seed run and independent real-data validation if results support it.
