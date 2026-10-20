# ASL identifiability-aware experiment report

**Run status:** EXPLORATORY OPTIMIZED TRAINING RUN.

## Question and simulator
Tests whether local information in multi-PLD ASL signals predicts ATT estimation errors and whether timing-conditioned spatial estimation changes errors. The forward equation and SCALE=100000 follow the reference implementation. The observation is a sum of two Rician magnitudes. High-SNR signal consistency uses the explicitly approximate `2 * noiseless_signal`; it is not an exact likelihood.

## Data and models
Independent synthetic brain phantoms were generated at 64x64. Training/validation/test phantom IDs are disjoint. Normalization was fitted on valid training voxels only. Baseline is the reference-style four-channel spatial U-Net with widths [16, 32, 64, 128]; IA-ASLNet uses shared per-PLD features and actual PLD conditioning. Full reference width is [32,64,128,256]. Smoke-width/model and short epochs are not baseline reproduction. Optimized mode trains against masked standardized MSE; smoke/full use the reference-style masked L1.

## Training and evaluation
Device: cuda; train/validation/test counts: 480/80/30. Metrics are computed per phantom inside the valid tissue mask and then aggregated across phantoms. No clipping is used. Locked stress conditions were generated after training and not used for selection.

## Results
           att_rmse   att_mae  att_bias  att_pearson_r   att_ccc  cbf_rmse   cbf_mae  cbf_bias  cbf_pearson_r   cbf_ccc  inference_seconds_per_phantom  parameter_count
model                                                                                                                                                                  
baseline   0.054892  0.039236  0.002612       0.990853  0.987075  1.092551  0.812621 -0.026854       0.997385  0.996773                       0.003950        1942738.0
ia_timing  0.054775  0.039545  0.005124       0.992720  0.988242  1.047289  0.772097  0.186719       0.997671  0.996976                       0.003945        1949908.0

IA minus baseline mean ATT RMSE (seconds): -0.000117. IA timing model had lower mean ATT RMSE on this run.
Paired phantom bootstrap intervals are saved in `paired_phantom_bootstrap_ci.csv`; this resamples whole phantoms, not pixels.
The paired phantom bootstrap interval includes zero; this run does not establish an ATT improvement.
The highest per-phantom ATT RMSE occurred for `test-7-5` under `baseline`: 0.1111 s.

## Ablations
Ablations were not enabled for this run; no ablation result is claimed.

## Identifiability
The local information analysis scales CBF and ATT by their full configured ranges and uses a high-SNR Gaussian variance approximation (`2 sigma^2`) to the sum-of-Rician noise. Error association is computed at phantom level and is descriptive at smoke sample sizes. See `identifiability_timing_analysis.csv` and `test_phantom_identifiability.csv`.
Phantom-level Spearman rho=-0.5315, p=0.002505, n=30; nuisance-adjusted linear correlation=-0.3413 (SNR, mean ATT/CBF, GM fraction).

## Stress tests and caveats
See `stress_test_per_phantom.csv`. Synthetic generator assumptions, 2-D morphology, known noise calibration, limited sample size, and no real ASL-MRI validation limit any conclusion. The smoke run is a pipeline check, not a final scientific claim. The next justified experiment is a larger matched multi-seed run and independent real-data validation if results support it.
