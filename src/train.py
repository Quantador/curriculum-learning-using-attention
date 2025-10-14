import os
import math
import argparse
import yaml
from transformers import (
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
)
from src.tokenizer import get_tokenizer
from src.data import build_tokenized_dataset
from src.lm import build_model

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    cfg_ds = load_yaml(args.dataset_cfg)
    cfg_model = load_yaml(args.model_cfg)
    cfg_train = load_yaml(args.train_cfg)

    set_seed(42)
    tokenizer = get_tokenizer()

    train_ds, val_ds, _ = build_tokenized_dataset(cfg_ds, tokenizer, num_proc=cfg_ds.get("num_proc", 4))

    model = build_model(cfg_model, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # causal LM
    )

    os.makedirs(cfg_train["output_dir"], exist_ok=True)

    training_args = TrainingArguments(
        output_dir=cfg_train["output_dir"],
        per_device_train_batch_size=cfg_train["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg_train["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg_train["gradient_accumulation_steps"],
        learning_rate=cfg_train["learning_rate"],
        weight_decay=cfg_train["weight_decay"],
        warmup_steps=cfg_train["warmup_steps"],
        num_train_epochs=cfg_train["num_train_epochs"],
        lr_scheduler_type=cfg_train["lr_scheduler_type"],
        logging_steps=cfg_train["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg_train["eval_steps"],
        save_steps=cfg_train["save_steps"],
        save_total_limit=2,
        report_to="none",  # set "wandb" if you use W&B
        fp16=cfg_train.get("fp16", False),
        bf16=cfg_train.get("bf16", False),
        dataloader_num_workers=cfg_ds.get("num_proc", 4),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
    print(f"Validation perplexity: {ppl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_cfg", default="configs/dataset.yaml")
    parser.add_argument("--model_cfg", default="configs/model.yaml")
    parser.add_argument("--train_cfg", default="configs/train.yaml")
    main(parser.parse_args())
