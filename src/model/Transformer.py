from datasets import load_dataset
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.BPEDataset import load_tokenizer, BPEDataset
from src.model.Coders import EncoderStack, DecoderStack
from src.train import init_weights


class Transformer(nn.Module):
  def __init__(self, num_heads: int, dim: int, oupt_dim: int, n_hidn: int, bias: bool = False):
    super(Transformer, self).__init__()
    self.num_heads, self.bias, self.n_hidn = num_heads, bias, n_hidn
    self.dim, self.oupt_dim = dim, oupt_dim

    # define encoder layers
    self.e1 = EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias)
    self.e2 = EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias)
    self.e3 = EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias)

    # define decoder layers
    self.d1 = DecoderStack(dim=self.dim, n_hidn=self.dim * 2, num_heads=self.num_heads, bias=self.bias)
    self.d1 = DecoderStack(dim=self.dim, n_hidn=self.dim * 2, num_heads=self.num_heads, bias=self.bias)
    self.d3 = DecoderStack(dim=self.dim, n_hidn=self.dim * 2, num_heads=self.num_heads, bias=self.bias)

    # embedding, fully connected, and positional encoding layers
    self.fc = nn.Linear(self.dim, self.oupt_dim, bias=self.bias)
    self.softmax = nn.Softmax(dim=-1)
  # __init__()

  def forward(self, input, output):
    input = self.e1(input)
    input = self.e2(input)
    input = self.e3(input)

    output = self.d1(input, output)
    output = self.d2(input, output)
    output = self.d3(input, output)

    return self.softmax(self.fc(output))
  # forward()
# Transformer

if __name__ == "__main__":
  from config import CONFIG

  # Initialize device and model parameters
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Load dataset and tokenizer
  dataset = load_dataset("imdb")["train"]
  tokenizer, pretrained_model = load_tokenizer("../tokenizer.json", PRETRAINED_MODEL=CONFIG["pretrained_model"])
  tokenizer.enable_truncation(max_length=CONFIG["dim"])
  tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=CONFIG["dim"])
  trainset = BPEDataset(dataset=dataset, encode=(tokenizer, pretrained_model), dim=CONFIG["dim"])

  # Initialize model, criterion, and optimizer
  model = Transformer(num_heads=CONFIG["num_heads"], dim=CONFIG["dim"], bias=False, oupt_dim=len(trainset.num_classes), n_hidn=2)
  model.to(device)
  model.apply(init_weights)
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
      progress.set_postfix({"loss": loss.item()})
  # for for
# if __name__ == "__main__":