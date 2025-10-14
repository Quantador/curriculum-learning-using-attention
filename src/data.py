import token
from datasets import load_dataset
from functools import partial



def tokenize(examples, tokenizer):
    return tokenizer(examples["text"])

def group_texts(examples, block_size):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def build_tokenized_dataset(cfg_ds, tokenizer, num_proc=4):
    raw = load_dataset(cfg_ds["name"], cfg_ds["config"])
    tokenized = raw.map(
        partial(tokenize, tokenizer=tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=raw["train"].column_names,
    )
    lm_dataset = tokenized.map(
        partial(group_texts, block_size=cfg_ds["seq_len"]),
        batched=True,
        num_proc=num_proc,
    )
    return lm_dataset["train"], lm_dataset["validation"], lm_dataset.get("test", None)