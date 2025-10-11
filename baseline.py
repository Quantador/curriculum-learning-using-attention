import math, os, random
import datasets
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, GPT2Config, GPT2LMHeadModel,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer
)
import evaluate
import numpy as np


SEED = 1337
random.seed(SEED); np.random.seed(SEED)

BLOCK_SIZE = 512

ds = load_dataset("wikitext", "wikitext-2-raw-v1")  # train/validation/test
tokenizer = AutoTokenizer.from_pretrained("gpt2")   # 50k BPE; add pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    loss_fct = model.get_output_embeddings().weight.new_zeros(1)  # dummy to get device
    # use HF’s built-in loss via Trainer, here we just report ppl from eval_loss
    return {}

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=BLOCK_SIZE, return_overflowing_tokens=False)

# Concatenate & chunk into 512-token blocks for causal LM
def group_texts(examples):
    # flatten
    ids = [id for seq in examples["input_ids"] for id in seq]
    # drop remainder
    total_len = (len(ids) // BLOCK_SIZE) * BLOCK_SIZE
    ids = ids[:total_len]
    # chunk
    chunks = [ids[i:i+BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
    return {"input_ids": chunks, "labels": chunks.copy(), "attention_mask": [[1]*BLOCK_SIZE]*len(chunks)}


def main():

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    lm_datasets = tokenized.map(group_texts, batched=True)

    # 2) Model (≈45–60M params)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=BLOCK_SIZE,
        n_ctx=BLOCK_SIZE,
        n_layer=12,
        n_head=8,
        n_embd=512,
        n_inner=2048,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    # 3) Collator (no masked LM; pure causal LM)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4) Metrics (perplexity)
    metric = evaluate.load("perplexity")  # optional, we’ll also compute manually

    args = TrainingArguments(
        output_dir="runs/baseline_wt2",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,        # effective batch ~= 32 sequences
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=2_000,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        eval_steps=2_000,
        logging_steps=200,
        save_steps=2_000,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=2,
        report_to=["wandb"],
        push_to_hub=True,
        hub_model_id="gpt2-wt2-baseline",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )


    trainer.train()
    eval_out = trainer.evaluate()
    eval_loss = eval_out["eval_loss"]
    print(f"Validation loss: {eval_loss:.4f} | Perplexity: {math.exp(eval_loss):.2f}")



if __name__ == "__main__":
    main()
