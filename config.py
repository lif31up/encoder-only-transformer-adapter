from datasets import load_dataset
from torch import nn
from transformers import BertTokenizer, BertModel, pipeline, AutoModelForSequenceClassification, AutoTokenizer
from EmbeddedDataset import embed


class Config:
  def __init__(self):
    self.n_heads = 12
    self.n_stacks = 1
    self.n_hidden = 2
    self.dim = 768
    self.output_dim = 2
    self.bias = True

    self.dropout = 0.1
    self.attention_dropout = 0.1
    self.eps = 1e-3
    self.betas = (0.9, 0.98)
    self.epochs = 5
    self.batch_size = 16
    self.lr = 1e-4
    self.clip_grad = False
    self.mask_prob = 0.3
    self.init_weights = init_weights

    self.pretrained_model = "bert-base-uncased"
    self.textset, self.testset_for_test = get_textset()
    self.save_to = "/content/drive/MyDrive/Colab Notebooks/BERT.bin"
    self.embedder, self.tokenizer = get_embedder(self.pretrained_model) # you can change the tokenizer setting on `./tokenizer.json`.
    self.dummy = None
  # __init__
# Config

def get_embedder(pretrained_model, distill=False):
  tokenizer = BertTokenizer.from_pretrained(pretrained_model)
  tokenizer.truncation_side = "right"
  tokenizer.padding_side = "right"
  return BertModel.from_pretrained(pretrained_model, output_hidden_states=False if distill else True, output_attentions=False), tokenizer
# get_embedder

def get_textset():
  dataset = load_dataset('imdb')['train']
  textset = dataset.shuffle(seed=42).select(range(100))
  textset_for_test = dataset.shuffle(seed=42).select(range(100, 201))
  return textset, textset_for_test
# get_textset

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)
# init_weights