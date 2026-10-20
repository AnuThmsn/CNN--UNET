# CNN-UNET ASL experiment

This project contains the standalone notebook [`notebooks/ASL_Identifiability_Aware_Architecture.ipynb`](notebooks/ASL_Identifiability_Aware_Architecture.ipynb). It simulates 2-D multi-PLD ASL phantoms, verifies the audited forward model, trains a reference-style U-Net and an experimental timing-conditioned model, evaluates held-out and shifted phantoms, and saves actual run outputs under `outputs/ia_aslnet_experiment/`.

The default is a short **smoke test**, not a scientific performance claim. The architecture is a research prototype; the notebook does not presume it is novel or superior.

## Windows setup (VS Code)

From PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name asl-cnn-unet --display-name "Python (ASL CNN-UNET)"
```

Open this folder in VS Code, open the notebook, and select the `Python (ASL CNN-UNET)` kernel. The notebook detects CUDA and prints the selected device and GPU. For a CUDA-enabled PyTorch build, install the version matching the local NVIDIA driver using the official PyTorch installation selector before installing the remaining requirements. No package installation or restart is triggered by notebook cells.

Run cells top-to-bottom. Keep the current directory at this project root so output paths resolve beneath this folder. For a larger exploratory run, set `CONFIG["mode"]` to `"optimized"`: it trains on more phantoms for longer and uses masked standardized MSE with CBF weighted 4:1 against ATT. This is an exploratory RMSE-oriented objective, not an exact reference reproduction. A recorded follow-up experiment additionally weighted training toward SNR 10; its runtime settings, checkpoints, and fresh held-out metrics are under `outputs/ia_aslnet_optimized_v3_cbfweighted/`. For the reference-width experiment, set `"full"`; it is computationally expensive, so reduce `batch_size` first if CUDA reports out-of-memory. Best and latest-epoch checkpoints are saved for validation recovery and resumption. Set `run_three_seeds=True` for seeds 42, 123, and 2024. `run_ablation=True` enables the optional guidance ablations. Do not merge smoke and full runs.

## Model and signal conventions

```text
Baseline:
4 observed PLD maps -> spatial U-Net encoder -> bottleneck
                    -> skip-connected decoder -> CBF / ATT maps

IA-ASLNet prototype:
L signal maps + their actual PLD values
    -> shared per-PLD CNN -> mean/max across PLDs
    -> timing FiLM -> spatial U-Net -> CBF / ATT maps
```

The reference signal constants and forward equation are embedded in the notebook (`SCALE=100000`). Its observation is the sum of two independently noisy Rician magnitudes, whose zero-noise/high-SNR signal is approximately twice the underlying signal. It is not control-minus-label subtraction. The optional physics consistency term uses a high-SNR signal-domain approximation and is off by default; it is not an exact Rician-sum likelihood.

## Reference audit and limitations

The notebook was constructed after inspecting `data_gen.py`, `experiments_unet_4pld/config.json`, `generate_spatial_data.py`, `dataset.py`, `model.py`, `train.py`, and the initial audit in [the reference repository](https://github.com/AnuThmsn/DNN-based-ASL-parameter-estimation-). The original repository is not imported or modified. The phantom generator follows its spatial assumptions, while smoke-width/fewer-epoch runs are not baseline reproductions. All data are synthetic 2-D phantoms; actual ASL-MRI validation remains necessary.

`build_notebook.py` regenerates the notebook from the embedded cell source. The notebook itself is self-contained and does not depend on that script at execution time.
