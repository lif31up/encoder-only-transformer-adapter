import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from src.config import CONFIG
from src.BPEDataset import BPEDataset
from src.model.BERT import BERT

def train(dataset, config=CONFIG, SAVE_TO="model"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer_config, model_config = CONFIG["tokenizer_config"], CONFIG["model"]
  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bert_tokenizer.truncation_side = "right"
  bert_tokenizer.padding_side = "right"
  bert_tokenizer.pad_token = bert_tokenizer.eos_token = "[PAD]"
  bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
  trainset = BPEDataset(dataset=dataset, dim=model_config["dim"], tokenizer=bert_tokenizer, model=bert_model)

  def init_weights(m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None: nn.init.zeros_(m.bias)
  # init_weights()

  if model_config["type"] == "BERT":
    model = BERT(model_config, init_weights=init_weights).to(device)
  else:
    return 1

  # Initialize model, criterion, and optimizer
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98))
  scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 1e-3))

  # Training loop
  progress = tqdm(range(config["iterations"]), desc="training", unit="epoch", leave=True, dynamic_ncols=True)
  for _ in progress:
    for feature, label in DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      output = model.forward(x=feature)
      loss = criterion(output, label)
      optim.zero_grad()
      loss.backward()
      if config["clip_grad"]: nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
      optim.step()
      scheduler.step()
      progress.set_postfix(loss=f"{loss.item():.4f}")
  # for for

  # saving model
  features = {
    "state": model.state_dict(),
    "config": config,
  }  # feature
  torch.save(features, f"{SAVE_TO}.pth")
  save_file(model.state_dict(), f"{SAVE_TO}.safetensors")
# train

if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  train(dataset, config=CONFIG, SAVE_TO="BERT")  # Replace with the actual model path
# __name__