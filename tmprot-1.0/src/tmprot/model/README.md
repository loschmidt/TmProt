---
base_model: facebook/esm2_t33_650M_UR50D
library_name: peft
tags:
  - protein
  - esm2
  - regression
  - thermostability
  - LoRA
  - peft
license: lgpl-3.0
---

# ESM-2 LoRA protein melting-temperature predictor

This directory contains the **six-epoch, seed-42** adapter for predicting protein melting temperature (Tm) from an amino-acid sequence. It is a PEFT LoRA adapter for [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D), not a standalone copy of the base model. The adapter weights are in `adapter_model.safetensors`; `adapter_config.json` specifies how they attach to the base model.

| Item | Value |
| --- | --- |
| Training configuration | Optuna trial 21 hyperparameters, six epochs, seed 42 |
| Checkpoint | Epoch 6, step 39330 |
| Adapter SHA-256 | `fedfe6b68ec0116e63050540bfc8cc1b3ea7b9df3193e9bb96c1286f120f0dbe` |
| Base model | ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) |
| LoRA | Rank 1, alpha 1, dropout 0.3616326822; query, key, and value projections |
| Output | Predicted Tm in °C |
| Input limit | 1022 amino-acid residues (1024 tokens including special tokens); longer sequences are truncated |

The model was trained on the Meltome-derived training split; ProTherm was used as an external evaluation set, not as training data. The regression head was trained alongside the adapter. Hyperparameters were chosen using Meltome validation RMSE; the selected configuration was then trained for six epochs with seeds 42, 43, and 44. **This directory contains seed 42 only.** The research results that average across three seeds require the other two adapters as well.

## Training parameters

| Parameter | Value |
| --- | --- |
| Learning rate | `0.0009802087957764465` |
| Schedule | Linear |
| Warm-up ratio | `0.06777238177111848` |
| Weight decay | `1.0354522925338798e-06` |
| Gradient clipping (`max_grad_norm`) | `0.14760402687280133` |
| Batch size | 4 |
| Epochs | 6 |
| Seed for this adapter | 42 |
| Training precision | Mixed bf16 |
| Optimizer | AdamW |
| Gradient checkpointing | Enabled |
| Maximum input length | 1024 tokens |

The values above come from the trial-21 configuration and six-epoch training record. A `training_args.bin` file is not needed to load this adapter.

## Evaluation

For this seed-42 checkpoint, the saved Meltome test predictions give RMSE **5.565 °C**, PCC **0.770**, and SCC **0.573**. The trainer-logged epoch-6 Meltome validation RMSE is **5.703 °C**. These are single-seed values; they should not be presented as the three-seed manuscript mean. Independent evaluation sets have different assay conditions and show variable, often larger errors. Consult the [research repository](https://github.com/loschmidt/TmProt) for the full evaluation and per-set results.

Predictions do not account for pH, salt, ions, post-translational modifications, or other experimental conditions. The model truncates sequences above the input limit. Use experimental measurements to confirm decisions that depend on a predicted Tm.

## Use

Load the ESM-2 base model and attach this directory as a PEFT adapter. The app in this repository does so through `helpers.load_model`. It accepts FASTA input and reports predicted Tm for each protein.

The adapter is licensed under LGPL-3.0. Developed by Loschmidt Laboratories, Masaryk University.
