import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.BPEDataset import BPEDataset
from src.model.BERTxGPT import BERT

def evaluate(MODEL, dataset):
  data = torch.load(MODEL)
  config = data["config"]

  tokenizer_config, model_config = config["tokenizer_config"], config["model"]
  try:
    tokenizer, pretrained_model = load_tokenizer("tokenizer.json", tokenizer_config, model_config["dim"])
  except:
    tokenizer, pretrained_model = init_tokenizer(dataset["text"], "tokenizer.json", tokenizer_config)
  trainset = BPEDataset(dataset=dataset, encode=(tokenizer, pretrained_model), dim=model_config["dim"])

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = BERTxGPT(model_config)
  model.load_state_dict(data["state"])
  model.eval()

  # Training loop
  correct = 0
  for feature, label in tqdm(DataLoader(trainset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    output = model.forward(input=feature)
    if torch.argmax(output, dim=-1) == torch.argmax(label, dim=-1): correct += 1
  # for
  print(f"Accuracy: {correct / len(trainset):.4f}")
# train

if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  evaluate("BERT.pth", dataset)  # Replace with the actual model path
# __name__