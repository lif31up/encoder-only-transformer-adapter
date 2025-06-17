import torch
from datasets import load_dataset
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CONFIG
from src.BPEDataset import BPEDataset, create_tokenizer, load_tokenizer
from src.model.BERT import BERT
from src.model.GPT import GPT

def train(model, dataset, tokenizer, SAVE_TO="model", clip_grad=False):
  # Initialize tokenizer and dataset
  tokenizer.enable_truncation(max_length=CONFIG["dim"])
  tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=CONFIG["dim"])
  trainset = BPEDataset(dataset=dataset, encode=(tokenizer, pretrained_model), dim=CONFIG["dim"])

  # Initialize model, criterion, and optimizer
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.98))
  scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 1e-3))

  # Training loop
  progress = tqdm(range(CONFIG["iterations"]), desc="training", unit="epoch", leave=True, dynamic_ncols=True)
  for _ in progress:
    for feature, label in DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      output = model.forward(input=feature)
      loss = criterion(output, label)
      optim.zero_grad()
      loss.backward()
      if clip_grad: nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
      optim.step()
      progress.set_postfix(loss=f"{loss.item():.4f}")
  # for for

  # saving model
  features = {
    "state": model.state_dict(),
    "config": CONFIG,
  }  # feature
  torch.save(features, f"{SAVE_TO}.pth")
  save_file(model.state_dict(), f"{SAVE_TO}.safetensors")
# train

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
      nn.init.zeros_(m.bias)
# init_weights()

if __name__ == "__main__":
  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  create_tokenizer(dataset["text"], "tokenizer.json")
  tokenizer, pretrained_model = load_tokenizer("tokenizer.json", PRETRAINED_MODEL=CONFIG["pretrained_model"])

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  vurt = BERT(num_heads=CONFIG["num_heads"], dim=CONFIG["dim"], oupt_dim=len(set(dataset["label"])), n_hidn=2, bias=False, n_stack=3).to(device)
  pity = GPT(num_heads=CONFIG["num_heads"], dim=CONFIG["dim"], oupt_dim=len(set(dataset["label"])), n_hidn=2, bias=False, n_stack=3).to(device)

  train(vurt, dataset, tokenizer, SAVE_TO="bert_model", clip_grad=True)
  train(pity, dataset, tokenizer, SAVE_TO="gpt_model", clip_grad=True)
# __name__