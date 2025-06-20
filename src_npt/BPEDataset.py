import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch.nn.functional as F

class BPEDataset(Dataset):
  def __init__(self, dataset, tokenizer, dim):
    self.dataset, self.tokenizer = dataset, tokenizer
    self.num_classes, self.dim = list(set(dataset["label"])), dim
  # __init__()

  def __len__(self): return len(self.dataset)

  def __getitem__(self, item):
    assert item < len(self.dataset), "Index out of bounds"
    feature, label = self.dataset["text"][item], self.dataset["label"][item]
    label = F.one_hot(torch.tensor(self.num_classes.index(label)), num_classes=len(self.num_classes)).float()
    feature = embed_without_pt(feature, self.tokenizer, max_length=self.dim)
    return positional_encode(feature), label
# Dataset

def embed_without_pt(text: str, tokenizer, max_length):
  return tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=528)["input_ids"].squeeze(0).float()
# embed(): encode input using `bert-base-uncased` tokenizer

def positional_encode(tensor):
  for i, value in enumerate(tensor):
    if i % 2 == 0: # even number
      tensor[i] = torch.sin(value / (10000 ** ((2 * i) / tensor.shape[-1])))
    else:
      tensor[i] = torch.cos(value / (10000 ** ((2 * i) / tensor.shape[-1])))
  return tensor
# positional encode(): return positional encoded input

if __name__ == "__main__":
  from datasets import load_dataset
  from torch.utils.data import DataLoader
  from src.config import CONFIG

  # load the config
  tokenizer_config, model_config = CONFIG["tokenizer_config"], CONFIG["model"]

  # init a tokenizer, not pretrained model
  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bert_tokenizer.truncation_side = "right"
  bert_tokenizer.padding_side = "right"
  bert_tokenizer.pad_token = bert_tokenizer.eos_token = "[PAD]"

  # create a trainset
  trainset = BPEDataset(dataset=dataset, dim=model_config["dim"], tokenizer=bert_tokenizer)

  # simulate a training loop
  for feature, label in DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True, num_workers=0):
    print(f"Feature shape: {feature.shape}, Label shape: {label.shape}")
    print(f"Feature: {feature}")
    break  # Just to check the first batch
  # for
# if __name__ == "__main__":