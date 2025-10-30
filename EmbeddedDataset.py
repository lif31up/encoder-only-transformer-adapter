import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class EmbeddedDataset(Dataset):
  def __init__(self, dataset, embedder, model, dim):
    self.dataset, self.tokenizer, self.model = dataset, embedder, model
    self.num_classes, self.dim = list(set(dataset['label'])), dim
    self.x, self.y = list(), list()
    self.is_consolidated = False
  # __init__

  def __len__(self): return len(self.dataset)

  def __getitem__(self, item):
    assert item < len(self.x) or item < len(self.y), "the index is out of bounds"
    assert self.is_consolidated is False, "the dataset isn't consolidated"
    return self.x[item], self.y[item]
  # __getitem__

  def consolidate(self):
    for feature, label in zip(self.dataset['text'], self.dataset['label']):
      label = F.one_hot(torch.tensor(self.num_classes.index(label)), num_classes=len(self.num_classes)).float()
      feature = embed(feature, self.tokenizer, self.model)
      self.y.append(label)
      self.x.append(feature)
    self.is_consolidated = True
  #consolidate
# EmbeddedDataset

def embed(text, tokenizer, model):
  text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
  with torch.no_grad(): output = model(**text)
  return output.last_hidden_state.squeeze(0)
# embed
