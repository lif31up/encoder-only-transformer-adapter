import torch
from datasets import load_dataset, tqdm
from torch import nn
from torch.utils.data import DataLoader
from config import CONFIG
from src.BPEDataset import BPEDataset, load_tokenizer
from src.model.AttentionNet import AttentionNet, init_weights

def train(TOKENIZER_PATH: str):
  # Initialize device and model parameters
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Load dataset and tokenizer
  dataset = load_dataset("imdb")["train"]
  tokenizer, pretrained_model = load_tokenizer(TOKENIZER_PATH, PRETRAINED_MODEL="bert-base-uncased")
  tokenizer.enable_truncation(max_length=CONFIG["dim"])
  tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
  trainset = BPEDataset(dataset=dataset, encode=(tokenizer, pretrained_model), dim=dim)

  # Initialize model, criterion, and optimizer
  model = AttentionNet(num_heads=num_heads, dim=trainset.dim, bias=bias, oupt_dim=len(trainset.num_classes)).to(device)
  model.apply(init_weights)
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.0001)

  # Training loop
  progress_bar = tqdm(range(iters), desc="Training", unit="epoch", leave=True, dynamic_ncols=True)
  for _ in progress_bar:
    for feature, label in DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      output = model.forward(feature).squeeze(1)
      loss = criterion(output, label)
      optim.zero_grad()
      loss.backward()
      optim.step()
      progress_bar.set_postfix({"loss": loss.item()})
# train()

if __name__ == "__main__": train(TOKENIZER_PATH="tokenizer.json")