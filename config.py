# CONFIGURATION
CONFIG = {
  "version": "1.0",
  "model": {
    "type": "BERT",
    "num_heads": 12,
    "dim": 768,  # = num_heads * 8
    "n_hidn": 2,
    "bias": False,
    "n_stack": 3,
    "oupt_dim": 2,  # Number of classes for classification
  },  # model_config
  "tokenizer_config": {
    "vocab_size": 30522,
    "special_tokens": ["[UNK]", "[PAD]", "[CLS]"],
    "pad_token": "[PAD]",
    "pretrained_model": "bert-base-uncased",
  }, # tokenizer_config
  "iterations": 10,
  "batch_size": 4,
  "learning_rate": 0.0001,
  "clip_grad": True,
} # CONFIG