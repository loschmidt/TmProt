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

# ESM-2 Protein Thermostability Predictor (LoRA Fine-Tuned)

This model is a parameter-efficient fine-tuned version of `facebook/esm2_t33_650M_UR50D` using the `PEFT` (`LoRA`) framework. The model is trained to predict protein thermostability (Tm) using the ProMelt dataset (combination of Meltome and ProTherm). The output is produced by a single neuron, albeit some modifications are planned such as MLP for Tm prediction. No additional fine-tuning using BRENDA was conducted. 

The model uses a single output neuron for regression, though future improvements (e.g., replacing with an MLP head) are planned. 


## Model Details

### Model Description
- **Base model:** facebook/esm2_t33_650M_UR50D (650M parameters)

- **Fine-tuning method:** LoRA (Low-Rank Adaptation) using PEFT

- **Task:** Protein thermostability prediction (regression)

- **Data:** ProMelt dataset (train/val/test CSV files)

- **Output layer:** Single linear regression head

- **Library stack:** Hugging Face Transformers, PEFT, PyTorch, Accelerate, MLflow, DagsHub

### Model Features

- Parameter-efficient fine-tuning (LoRA) for memory and compute savings

- Cosine learning rate schedule

- Mixed precision (fp16) training via Accelerate

- Early stopping and best model selection based on RMSE

- Automatic MLflow logging and artifact tracking

### Additional details

- **Developed by:** Loschmidt Laboratories
- **Model type:** Protein sequence regression model (ESM-2 backbone + LoRA adapter)
- **Language(s) (NLP):** Protein sequences (amino acids as chars)
- **License:** This project is licensed under the GNU Lesser General Public License v3.0.
- **Finetuned from model:** facebook/esm2_t33_650M_UR50D

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [\[LL repo\]](https://git.loschmidt.cz/tmprot/tmprot-predictor)
- **Paper [optional]:** [In progress]
- **Demo [optional]:** [In progress]

## Usage

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use
```
pip install -r requirements.txt
cd src/tmprot
python cli.py -f ../../test/FIR.fasta  -s ../../predictions/
```

### Out-of-Scope Use
The generated $Tm$-aware embeddings from optimized ESM2 model can be used as features for MLPRegressor. 

## Bias, Risks, and Limitations
Predictions do not generalize well outside the proteomics-based ProMelt dataset, thus the results on the independent sets are worse.
Additionally:

- It does not account for post-translational modifications or environmental factors (e.g., pH, salt, ions).

### Recommendations
- Use outputs in combination with experimental or domain expertise.

- Consider ensemble methods or downstream MLP for robustness.

## How to Get Started with the Model
Prepare a FASTA file with your protein(s).

Use the CLI to predict:

    python cli.py -f path/to/input.fasta -s path/to/output/directory

    The output is saved as a JSON list: [id, sequence, tm].

For code integration, use the TmPredictor class in `src/tmprot/cli.py`.
## Training Details

### Training Data

The model was trained on the ProMelt dataset — a curated combination of the Meltome Atlas and ProTherm datasets, containing protein sequences with experimentally measured melting temperatures using proteomics-based approaches. Sequences were filtered to remove duplicates and split into train/val/test sets with sequence identity = 25%. CSV were stored in `../data/promelt/`.

### Training Procedure
#### Preprocessing
- Sequence longer than 2000 AAs were filtered out.
- Sequences tokenized using ESM-2 tokenizer from Hugging Face Transformers.
- Batched using `DefaultDataCollator` with dynamic padding.
#### Training Hyperparameters

| Parameter              | Value                        |
|------------------------|------------------------------|
| Model                  | facebook/esm2_t33_650M_UR50D |
| LoRA rank              | 1                            |
| LoRA alpha             | 1                            |
| LoRA dropout           | 0.1                          |
| Learning rate          | 4.95e-4                      |
| Weight decay           | 1.56e-5                      |
| Gradient clipping      | 0.805                        |
| Batch size             | 16                           |
| Epochs                 | 1                            |
| Precision              | fp16 (mixed)                 |
| Scheduler              | Cosine                       |
| Optimizer              | AdamW                        |
| Evaluation strategy    | Per epoch                    |
| Save strategy          | Per epoch                    |
| Best model selection   | Based on RMSE                |
| Gradient checkpointing | Enabled                      |
| MLflow tracking        | Enabled (via DagsHub)        |


- LoRA target modules: query, key, and value

- Loss function: MSE loss (via Trainer for regression)

- Evaluation metrics: RMSE, R2, Pearson, Spearman
#### Speeds, Sizes, Times [optional]
- ~4200 seconds for training and evaluation.
- Inference speed: ~5 sec/protein 
- 7.3M size for `MODEl.pt` file with adapters and updated weights.

## Evaluation

The model was evaluated on training, validation, and test datasets using multiple regression metrics to assess performance in predicting protein thermostability (Tm). Evaluation was performed after training for one epoch, with early stopping based on the validation RMSE.
### Testing Data, Factors & Metrics

#### Testing Data

The test set consists of ~7300 proteins held out from ProMelt. Care was taken to ensure no >25% sequence identity with training samples.

#### Factors

[More Information Needed]

#### Metrics
- RMSE (Root Mean Square Error): Measures average prediction error magnitude.

- R2 Score (Coefficient of Determination): Indicates the proportion of variance explained by the model.

- PCC (Pearson's Correlation Coefficient): Measures linear correlation between predicted and actual Tm values.

- SCC (Spearman's Correlation Coefficient): Measures monotonic relationship between predicted and actual Tm values.

### Results

| Metric                    | Train          | Validation    | Test          |
|---------------------------|----------------|---------------|---------------|
| Loss                      | 31.14          | 34.94         | 39.48         |
| RMSE                      | 5.58           | 5.91          | 6.28          |
| R2 Score                  | 0.685          | 0.656         | 0.687         |
| Pearson Correlation (PCC) | 0.828          | 0.810         | 0.830         |
| Spearman Correlation (SCC)| 0.635          | 0.585         | 0.617         |
| Runtime (seconds)         | 1602.62        | 178.19        | 337.08        |
| Samples per second        | 21.44          | 21.45         | 21.45         |
| Steps per second          | 5.36           | 5.37          | 5.36          |
| Epoch                     | 1              | 1             | 1             |

These metrics indicate that the model achieves good regression performance on the protein thermostability prediction task, with reasonable generalization from training to test data.

#### Summary

This model is a LoRA fine-tuned version of the ESM-2 PLM (facebook/esm2_t33_650M_UR50D) designed to predict protein thermostability (Tm) from sequence data. The training was conducted on the ProMelt dataset with a single output regression head. Evaluation shows consistent performance across training, validation, and test splits with RMSE around 5.6-6.3 and good correlation metrics (R2 ~0.65-0.69, PCC ~0.81-0.83). This model provides a lightweight, efficient solution for protein thermostability prediction with potential applications in protein engineering and stability screening.

---

## Model Examination [optional]

Interpretability analyses for this model remain to be conducted. Future work may include:

- Visualization of attention maps to identify sequence regions most relevant for thermostability.
- Embedding space analysis to examine clustering of proteins by thermostability.

  
These studies will help illuminate how the LoRA adapters modulate the ESM-2 backbone to capture thermostability-related features.

---
## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** 10 GB part (MIG) A100
- **Hours used:** ?
- **Cloud Provider:** Metacentrum
- **Compute Region:** Czech republic
- **Carbon Emitted:** ?

The use of LoRA parameter-efficient fine-tuning significantly reduces training time and energy consumption compared to full model fine-tuning, contributing to lower carbon footprint.
## Technical Specifications [optional]

### Model Architecture and Objective

- **Backbone:** ESM-2 PLM with 650 million parameters  
- **Fine-tuning:** LoRA adapters applied to attention query, key, and value modules  
- **Output:** Single linear regression head predicting protein melting temperature (Tm)  
- **Objective:** Minimize RMSE between predicted and measured Tm values

### Compute Infrastructure

Training utilized a single NVIDIA A100 GPU with mixed precision enabled via the Accelerate library to optimize memory and speed.


#### Hardware

- GPU: NVIDIA A100 10GB  
- RAM: 16 GB  



#### Software

- Python 3.9+
- PyTorch==2.5.1
- transformers==4.47.1
- pandas==2.2.3
- accelerate==1.1.1
- datasets==3.1.0
- peft==0.13.2
- scipy==1.14.1
- scikit-learn==1.5.2
- prettytable==3.12.0
- mlflow==2.18.0
- dagshub (latest stable)
- optuna (latest stable)
- seaborn==0.13.2


## Citation [optional]
Paper: In progress. A manuscript detailing this model's methodology and performance is currently being prepared and will be linked here once published.

**BibTeX:**

[TODO]

**APA:**

[TODO]

## Glossary [optional]
Tm (Melting Temperature): The temperature at which half of the protein denatures.

LoRA (Low-Rank Adaptation): A parameter-efficient fine-tuning method that inserts trainable rank-decomposed matrices into each layer of the transformer.

RMSE (Root Mean Squared Error): Common regression metric measuring average model prediction error.

PCC (Pearson Correlation Coefficient): Measures the linear correlation between predicted and true values.

SCC (Spearman Correlation Coefficient): Measures the rank correlation between predicted and true values.

fp16 (Mixed Precision): A technique that uses 16-bit floating point numbers for faster and more memory-efficient training.

## More Information [optional]

For additional details, updates, and community discussion:

Repository: https://git.loschmidt.cz/tmprot/tmprot-predictor
## Model Card Authors [optional]
- pailozian@fnusa.cz
- add contacts ...

Loschmidt Laboratories (Masaryk University)
## Model Card Contact
Issue Tracker: https://git.loschmidt.cz/tmprot/tmprot-predictor/issues

### Framework versions

- PEFT 0.13.2