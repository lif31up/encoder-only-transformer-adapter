# CONFIGURATION
CONFIG = {
  "version": "1.0.1",
  "model": {
    "type": "BERT",
    "num_heads": 12,
    "dim": 768,
    "n_hidn": 2,
    "bias": False,
    "n_stack": 1,
    "oupt_dim": 2,
    "dropout": 0.1,
    "attention_dropout": 0.1,
  },  # model_config
  "tokenizer_config": {
    "vocab_size": 30522,
    "special_tokens": ["[UNK]", "[PAD]", "[CLS]"],
    "pad_token": "[PAD]",
    "pretrained_model": "bert-base-uncased",
  }, # tokenizer_config
  "epsilon": 1e-3,
  "epochs": 5,
  "batch_size": 2,
  "learning_rate": 1e-4,
  "clip_grad": False,
} # CONFIG