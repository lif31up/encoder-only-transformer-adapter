import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from EmbeddedDataset import EmbeddedDataset


def distill_train(model, path, config, trainset, device):
  model.to(device)
  cross_entropy_loss, kv_divergence_loss = nn.CrossEntropyLoss(), nn.KLDivLoss()
  optim = torch.optim.Adam(model.parameters(), lr=config.lr, eps=config.eps, betas=config.betas)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 1e-3))

  progress = tqdm(range(config.epochs), desc="TRAINING", unit="epoch", leave=True, dynamic_ncols=True)
  for _ in progress:
    for feature, (hard_label, soft_label) in DataLoader(
        dataset=trainset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4):
      feature, hard_label, soft_label = feature.to(device, non_blocking=True), hard_label.to(device, non_blocking=True), soft_label.to(device, non_blocking=True)
      output = model.forward(x=feature) / config.temperature
      soft_pred = output / config.temperature
      distill_loss = kv_divergence_loss(soft_pred, soft_label)
      student_loss = cross_entropy_loss(output, hard_label)
      total_loss = config.lr * distill_loss + (1 - config.lr) * student_loss
      optim.zero_grad()
      total_loss.backward()
      if config.clip_grad: nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optim.step()
      scheduler.step()
      progress.set_postfix(loss=f"{total_loss.item():.4f}")

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
  bert_config.distill = True
  trainset = EmbeddedDataset(dataset=bert_config.textset, dim=bert_config.dim, embedder=bert_config.embedder, model=bert_config.embedder)
  model = Transformer(bert_config)
  distill_train(model=model, path=bert_config.save_to, trainset=trainset, config=bert_config, device=device)
# if __name__ == "__main__":