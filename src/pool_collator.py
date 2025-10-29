import torch
from transformers import DataCollatorWithPadding

class PoolCollator:
    """
    Wrap a padding collator to produce candidate pools of size M.
    - Takes a list of individual examples from HF datasets
    - Groups them into a single dict with tensors of shape [M, seq_len]
    """
    def __init__(self, tokenizer, M: int):
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        self.M = M

    def __call__(self, features_list):
        # HF Trainer passes a list of length == per_device_train_batch_size
        # We want exactly M candidates in a pool, so enforce that via batch size config:
        # set per_device_train_batch_size = M
        batch = self.pad(features_list)
        # Now batch["input_ids"]: [M, Lmax]. For causal LM, labels = input_ids.
        batch["labels"] = batch["input_ids"].clone()
        return batch
