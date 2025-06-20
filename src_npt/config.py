# CONFIGURATION
CONFIG = {
  "version": "1.0",
  "model": {
    "type": "BERT",
    "num_heads": 12,
    "dim": 528,
    "n_hidn": 2,
    "bias": False,
    "n_stack": 6,
    "oupt_dim": 2,  # Number of classes for classification
  },  # model_config
  "tokenizer_config": {
    "vocab_size": 30522,
    "special_tokens": ["[UNK]", "[PAD]", "[CLS]"],
    "pad_token": "[PAD]",
    "pretrained_model": "bert-base-uncased",
  }, # tokenizer_config
  "iterations": 30,
  "batch_size": 32,
  "learning_rate": 3e-5,
  "clip_grad": True,
} # CONFIG