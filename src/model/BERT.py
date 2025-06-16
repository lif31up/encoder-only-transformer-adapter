from safetensors.torch import save_file
from src.model.Coders import EncoderStack
from datasets import load_dataset
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.BPEDataset import load_tokenizer, BPEDataset

class BERT(nn.Module):
  def __init__(self, num_heads: int, dim: int, oupt_dim: int, n_hidn: int, bias: bool = False):
    super(BERT, self).__init__()
    self.num_heads, self.bias, self.n_hidn = num_heads, bias, n_hidn
    self.dim, self.oupt_dim = dim, oupt_dim

    # define encoder layers
    self.e1 = EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias)
    self.e2 = EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias)
    self.e3 = EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias)

    # embedding and fully connected layers
    self.fc = nn.Linear(self.dim, self.oupt_dim, bias=self.bias)
    self.softmax = nn.Softmax(dim=-1)
  # __init__()

  def forward(self, input):
    input = self.e1(input)
    input = self.e2(input)
    input = self.e3(input)

    return self.softmax(self.fc(input))
  # forward()
# BERT

if __name__ == "__main__":
  from config import CONFIG

  # Initialize device and model parameters
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Load dataset and tokenizer
  dataset = load_dataset("imdb")["train"].shuffle(seed=42).select(range(100))
  tokenizer, pretrained_model = load_tokenizer("../tokenizer.json", PRETRAINED_MODEL=CONFIG["pretrained_model"])
  tokenizer.enable_truncation(max_length=CONFIG["dim"])
  tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=CONFIG["dim"])
  trainset = BPEDataset(dataset=dataset, encode=(tokenizer, pretrained_model), dim=CONFIG["dim"])

  # Initialize model, criterion, and optimizer
  data = torch.load("model.pth")
  model = BERT(num_heads=CONFIG["num_heads"], dim=CONFIG["dim"], bias=False, oupt_dim=len(trainset.num_classes), n_hidn=2).to(device)
  model.load_state_dict(data["state"])

  """
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.98))
  scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 1e-3))

  # Training loop
  progress = tqdm(range(CONFIG["iterations"]), desc="training", unit="epoch", leave=True, dynamic_ncols=True)
  for _ in progress:
    for feature, label in DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      output = model.forward(feature)
      loss = criterion(output, label)
      optim.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
      optim.step()
      progress.set_postfix(loss=f"{loss.item():.4f}")
  # for for

  # saving model
  features = {
    "state": model.state_dict(),
    "config": {
      "num_heads": CONFIG["num_heads"],
      "dim": CONFIG["dim"],
      "bias": False,
      "oupt_dim": len(trainset.num_classes),
      "n_hidn": 2
    }
  }  # feature
  torch.save(features, "model.pth")
  save_file(model.state_dict(), "model.safetensors")
  """

  total_accuracy, n_iters = [], 0
  for feature, label in DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    output = model.forward(feature)
    correct = 0
    for i in range(len(output)):
      if torch.argmax(label[i]) == torch.argmax(label[i]): correct += 1
    # for
    total_accuracy.append(correct / len(output))
  # for
  print(f"Total accuracy: {sum(total_accuracy) / len(total_accuracy):.2f}%")
# if __name__ == "__main__":