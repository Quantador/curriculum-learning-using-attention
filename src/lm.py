from transformers import GPT2Config, GPT2LMHeadModel

def build_model(cfg_model, vocab_size, pad_token_id):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=cfg_model["n_layer"],
        n_head=cfg_model["n_head"],
        n_positions=cfg_model["n_positions"],
        n_ctx=cfg_model["n_ctx"],
        n_embd=cfg_model["n_embd"],
        resid_pdrop=cfg_model["dropout"],
        embd_pdrop=cfg_model["dropout"],
        attn_pdrop=cfg_model["dropout"]
    )
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = pad_token_id
    return model
