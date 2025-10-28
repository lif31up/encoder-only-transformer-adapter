import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from EmbeddedDataset import EmbeddedDataset


def train(model, path, config, trainset, device):
  model.to(device)
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=config.lr, eps=config.eps, betas=config.betas)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 1e-3))

  progress = tqdm(range(config.epochs), desc="TRAINING", unit="epoch", leave=True, dynamic_ncols=True)
  for _ in progress:
    for feature, label in DataLoader(trainset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      output = model.forward(x=feature)
      loss = criterion(output, label)
      optim.zero_grad()
      loss.backward()
      if config.clip_grad: nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optim.step()
      scheduler.step()
      progress.set_postfix(loss=f"{loss.item():.4f}")

  # saving model
  features = {
    "state": model.state_dict(),
    "config": config,
  }  # feature
  torch.save(features, path)
# train

if __name__ == "__main__":
  from config import Config
  from model.Transformer import Transformer
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  bert_config = Config()
  trainset = EmbeddedDataset(dataset=bert_config.textset, dim=bert_config.dim, embedder=bert_config.embedder, model=bert_config.embedder)
  model = Transformer(bert_config)
  train(model=model, path=bert_config.save_to, trainset=trainset, config=bert_config, device=device)
# if __name__ == "__main__":