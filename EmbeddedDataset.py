import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class EmbeddedDataset(Dataset):
  def __init__(self, dataset, embedder, model, dim):
    self.dataset, self.tokenizer, self.model = dataset, embedder, model
    self.num_classes, self.dim = list(set(dataset['label'])), dim
  # __init__

  def __len__(self): return len(self.dataset)

  def __getitem__(self, item):
    assert item < len(self.dataset), "Index out of bounds"
    feature, label = self.dataset['text'][item], self.dataset['label'][item]
    label = F.one_hot(torch.tensor(self.num_classes.index(label)), num_classes=len(self.num_classes)).float()
    feature = embed(feature, self.tokenizer, self.model)
    return feature, label
  # __getitem__
# EmbeddedDataset

def embed(text: str, tokenizer, model):
  text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
  with torch.no_grad(): output = model(**text)
  return output.last_hidden_state.squeeze(0)
# embed