import torch
from tokenizers import Tokenizer, models, trainers
from transformers import AutoModel
from torch.utils.data import Dataset

class BPEDataset(Dataset):
  def __init__(self, dataset, transforms):
    self.dataset, self.transforms = dataset, transforms
    self.dim = self.__getitem__(0).shape[1]
  # __init__()

  def __getitem__(self, item):
    assert item < len(self.dataset), "Index out of bounds"
    returned_item = self.dataset[item]
    if self.transforms:
      for transform in self.transforms: returned_item = transform(returned_item)
    return returned_item
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

def embed(text: str, tokenizer: Tokenizer, model: AutoModel):
  input_ids = torch.tensor([tokenizer.encode(text).ids])
  with torch.no_grad(): outputs = model(input_ids)
  return torch.mean(outputs.last_hidden_state, dim=1)
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
    print(embed("your text here", tokenizer, model).shape)
  if tag == "dataset":
    from datasets import load_dataset
    dataset = load_dataset("imdb")["train"]["text"]
    tokenizer, pretrained_model = load_tokenizer("tokenizer.json", PRETRAINED_MODEL="bert-base-uncased")
    transforms = [lambda x: embed(x, tokenizer, pretrained_model)]
    trainset = BPEDataset(dataset=dataset, transforms=transforms)
    print(trainset[0])
# __name__