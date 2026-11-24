"""
Retrain the HPO-selected ESM2+LoRA configuration with a fixed recipe, optionally
on train+val, then score it on the Meltome test split and every independent
eval-set bucket.

Reproduces esm_lora_hpo.py's objective() exactly (same model build, same
TrainingArguments, same seed) from a best_trial.json -- the only differences
are that val can be folded into training (--train-val) and that nothing is
evaluated mid-training. With num_train_epochs=1 there is a single checkpoint,
so load_best_model_at_end would pick it anyway.

Run: python src/training/hpo/retrain_best.py <best_trial.json> <out_dir> [--train-val] [seed=43]
"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

import torch
import pandas as pd
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer, EsmForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
import json

from src.data.loading import load_filtered_dataset
from src.data.tokenization import create_dataset
from src.eval.io import save_metrics_to_json
from src.eval.metrics import compute_metrics, get_metrics
from src.training.hpo.esm_lora_hpo import evaluate_best_model
from src.training.hpo.utils import load_hpo_config
from src.utils.helpers import count_parameters, set_seed


def build_model(cfg, params):
    model = EsmForSequenceClassification.from_pretrained(cfg.model_name, num_labels=1)
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=params["r"],
        lora_alpha=params["r"],  # ratio fixed at 1, as in esm_lora_hpo.py
        target_modules=["query", "key", "value"],
        lora_dropout=params["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model.esm, 'contact_head'):
        del model.esm.contact_head
        state_dict = model.state_dict()
        for k in [k for k in state_dict.keys() if 'contact_head' in k]:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=True)
    return model


def score_meltome_test(cfg, model_dir, out_dir):
    from src.models.lora import load_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(cfg.model_name, model_dir, device)
    model.eval()
    df = load_filtered_dataset(os.path.join(cfg.data_dir, "test_meltome.csv"), True)
    dataset = create_dataset(
        tokenizer, df["sequence"].tolist(), df["labels"].tolist(),
        split="eval", max_length=int(cfg.get("max_length", 1024)), cache_dir=None,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=out_dir, report_to="none"),
        eval_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    results = trainer.predict(dataset)
    preds, labels = results.predictions.flatten(), results.label_ids
    m = dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], [float(v) for v in get_metrics(preds, labels)]))
    print(f"[Retrain] Meltome test: {m}")
    pd.DataFrame({"ProteinID": df["ProteinID"], "Tm_Actual": labels, "Tm_Predicted": preds}
                 ).to_csv(os.path.join(out_dir, "test.csv"), index=False)
    save_metrics_to_json(m, os.path.join(out_dir, "test.json"))


def main():
    # key=value arguments are config overrides consumed by load_hpo_config (e.g. seed=43),
    # not positional arguments.
    args = [a for a in sys.argv[1:] if not a.startswith("--") and "=" not in a]
    train_val = "--train-val" in sys.argv
    if len(args) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <best_trial.json> <out_dir> [--train-val] [key=value ...]")
    best_trial_path, out_dir = args
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_hpo_config("conf/hpo/esm_lora.yaml")
    with open(best_trial_path) as f:
        params = json.load(f)["params"]
    fx = cfg.fixed
    print(f"[Retrain] params={params} train_val={train_val}")

    set_seed(int(cfg.seed))

    train_df = load_filtered_dataset(os.path.join(cfg.data_dir, "train_meltome.csv"), True)
    if train_val:
        val_df = load_filtered_dataset(os.path.join(cfg.data_dir, "val_meltome.csv"), True)
        train_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"[Retrain] training on {len(train_df)} proteins")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = build_model(cfg, params)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    count_parameters(model)

    train_dataset = create_dataset(tokenizer, train_df['sequence'].tolist(),
                                   train_df['labels'].tolist(), split="train", cache_dir=None)

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "trainer"),
        learning_rate=params["lr"],
        lr_scheduler_type=params["lr_scheduler_type"],
        warmup_ratio=params["warmup_ratio"],
        per_device_train_batch_size=int(fx.per_device_train_batch_size),
        per_device_eval_batch_size=int(fx.per_device_train_batch_size),
        num_train_epochs=int(fx.num_train_epochs),
        weight_decay=params["weight_decay"],
        max_grad_norm=params["max_grad_norm"],
        torch_empty_cache_steps=int(fx.torch_empty_cache_steps),
        dataloader_num_workers=int(fx.dataloader_num_workers),
        dataloader_pin_memory=bool(fx.dataloader_pin_memory),
        gradient_checkpointing=bool(fx.gradient_checkpointing),
        eval_strategy="no",
        save_strategy="no",
        bf16=bool(fx.bf16),
        bf16_full_eval=bool(fx.bf16_full_eval),
        fp16=bool(fx.fp16),
        report_to=fx.report_to,
        logging_steps=int(fx.logging_steps),
        logging_first_step=bool(fx.logging_first_step),
        seed=int(cfg.seed),
        data_seed=int(cfg.seed),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model_dir = os.path.join(out_dir, "model")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    del trainer, model, accelerator, train_dataset
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    score_meltome_test(cfg, model_dir, out_dir)
    evaluate_best_model(cfg, model_dir, out_dir)


if __name__ == "__main__":
    main()
