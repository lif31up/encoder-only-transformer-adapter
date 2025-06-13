import torch
from tokenizers import Tokenizer, models, trainers
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class BPEDataset(Dataset):
  def __init__(self, dataset, encode):
    self.dataset, self.encode = dataset, encode
    self.num_classes = list(set(dataset["label"]))
    self.dim = self.__getitem__(0)[0].shape[1]
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
  return torch.mean(outputs.last_hidden_state, dim=1, dtype=torch.float32)
# embed(): Encodes a given text using the tokenizer and computes the mean of the last hidden state from the model's output.

if __name__ == "__main__":
  save_path, tag = "tokenizer.json", "dataset"
  if tag == "load_dataset":
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    create_tokenizer(dataset["train"]["text"], PATH=save_path)
    print(f"Tokenizer created and saved to {save_path}.")
  if tag == "load_tokenizer":
    tokenizer, model = load_tokenizer(save_path, "bert-base-uncased")
    print(embed("your text here", tokenizer, model))
  if tag == "dataset":
    from datasets import load_dataset
    dataset = load_dataset("imdb")["train"]
    tokenizer, pretrained_model = load_tokenizer("tokenizer.json", PRETRAINED_MODEL="bert-base-uncased")
    trainset = BPEDataset(dataset=dataset, encode=(tokenizer, pretrained_model))
    for feature, label in DataLoader(trainset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4):
      print(feature.shape, label.shape)
      break
# __name__