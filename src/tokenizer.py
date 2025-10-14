from transformers import AutoTokenizer


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
        