# TmProt 1.0

CLI tool for predicting protein melting temperature (Tm) using ESM2 fine-tuned with LoRA.

## Installation

```bash
cd tmprot-1.0
pip install -e .
```

## Usage

### Basic prediction
```bash
tmprot --input proteins.fasta --outdir ./predictions
```

### With custom threshold
```bash
tmprot --input proteins.fasta --outdir ./predictions --threshold 50.0
```

### With custom delimiter
```bash
tmprot --input proteins.fasta --outdir ./predictions --delimiter ","
```

### Options

- `--input, -i` (required): Path to FASTA file
- `--outdir, -o` (optional): Output directory for CSV results
- `--threshold, -t` (default: 60.0): Thermostability threshold in °C
- `--delimiter, -d` (default: tab): CSV delimiter

## Output

CSV file with columns:
- **Rank**: Descending order by Tm
- **ID**: Sequence identifier
- **Predicted Tm [°C]**: Melting temperature prediction
- **Thermostable**: Yes/No based on threshold

## Model

- **Base**: ESM2 (facebook/esm2_t33_650M_UR50D)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Training data**: ProMelt dataset (~45K sequences)
- **Task**: Regression prediction of Tm

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- PEFT ≥ 0.4
- BioPython ≥ 1.81
