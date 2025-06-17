import torch
from tokenizers import Tokenizer, models, trainers
from transformers import AutoModel
from torch.utils.data import Dataset
import torch.nn.functional as F
from config import CONFIG

class BPEDataset(Dataset):
  def __init__(self, dataset, encode, dim):
    self.dataset, self.encode = dataset, encode
    self.num_classes = list(set(dataset["label"]))
    self.dim = dim
  # __init__()

  def __len__(self): return len(self.dataset)

  def __getitem__(self, item):
    assert item < len(self.dataset), "Index out of bounds"
    feature, label = self.dataset["text"][item], self.dataset["label"][item]
    label = F.one_hot(torch.tensor(self.num_classes.index(label)), num_classes=len(self.num_classes)).float()
    if self.encode: feature = embed(feature, *self.encode)
    return feature, label
# Dataset

def create_tokenizer(DATASET, PATH: str):
  # init tokenizer
  tokenizer = Tokenizer(models.BPE())
  trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[PAD]", "[CLS]"])
  tokenizer.train_from_iterator(DATASET, trainer)  # Train on IMDb text

  # save tokenizer
  tokenizer.save(PATH)
  return tokenizer, AutoModel.from_pretrained("bert-base-uncased")
# init_tokenizer(): Initializes a Byte-Pair Encoding (BPE) tokenizer with a specified vocabulary size and special tokens.

def load_tokenizer(PATH: str, PRETRAINED_MODEL: str):
  return Tokenizer.from_file(PATH), AutoModel.from_pretrained(PRETRAINED_MODEL)
# load_tokenizer(): Loads a tokenizer from a specified file path and initializes a pre-trained model.

def embed(text: str, tokenizer: Tokenizer, pretrained_model: AutoModel):
  input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True).ids])
  with torch.no_grad(): outputs = pretrained_model(input_ids)
  return torch.mean(outputs.last_hidden_state, dim=-1, dtype=torch.float32).squeeze(0)
# embed(): Encodes a given text using the tokenizer and computes the mean of the last hidden state from the model's output.