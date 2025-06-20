import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import torch.nn.functional as F

class BPEDataset(Dataset):
  def __init__(self, dataset, tokenizer, model, dim):
    self.dataset, self.tokenizer, self.model = dataset, tokenizer, model
    self.num_classes, self.dim = list(set(dataset["label"])), dim
  # __init__()

  def __len__(self): return len(self.dataset)

  def __getitem__(self, item):
    assert item < len(self.dataset), "Index out of bounds"
    feature, label = self.dataset["text"][item], self.dataset["label"][item]
    label = F.one_hot(torch.tensor(self.num_classes.index(label)), num_classes=len(self.num_classes)).float()
    feature = embed(feature, self.tokenizer, self.model)
    return feature, label
# Dataset

def embed(text: str, tokenizer, model):
  input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
  with torch.no_grad(): output = model(**input)
  return output.last_hidden_state.squeeze(0)
# embed(): Encodes a given text using the tokenizer and computes the mean of the last hidden state from the model's output.

if __name__ == "__main__":
  from datasets import load_dataset
  from torch.utils.data import DataLoader
  from src.config import CONFIG

  # load the config
  tokenizer_config, model_config = CONFIG["tokenizer_config"], CONFIG["model"]

  # init a tokenizer and pretrained model
  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bert_tokenizer.truncation_side = "right"
  bert_tokenizer.padding_side = "right"
  bert_tokenizer.pad_token = bert_tokenizer.eos_token = "[PAD]"
  bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
  # create a trainset
  trainset = BPEDataset(dataset=dataset, dim=model_config["dim"], tokenizer=bert_tokenizer, model=bert_model)

  # simulate a training loop
  for feature, label in DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True, num_workers=0):
    print(f"Feature shape: {feature.shape}, Label shape: {label.shape}")
    print(f"Feature: {feature}")
    break  # Just to check the first batch
  # for
# if __name__ == "__main__":