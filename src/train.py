from datasets import load_dataset
from config import CONFIG
from src.BPEDataset import embed, BPEDataset, load_tokenizer
from src.model.AttentionNet import AttentionNet

def train(TOKENIZER_PATH: str):
  num_heads, bias = CONFIG["num_heads"], CONFIG["bias"]

  dataset = load_dataset("imdb")["train"]["text"]
  tokenizer, pretrained_model = load_tokenizer(TOKENIZER_PATH, PRETRAINED_MODEL="bert-base-uncased")
  transforms = [lambda x: embed(x, tokenizer, pretrained_model)]
  trainset = BPEDataset(dataset=dataset, transforms=transforms)

  model = AttentionNet(num_heads, trainset.dim, bias)
  print(model(trainset[0]))
# train()

if __name__ == "__main__": train(TOKENIZER_PATH="tokenizer.json")