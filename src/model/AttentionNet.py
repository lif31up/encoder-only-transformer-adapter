from datasets import load_dataset
import torch.optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.BPEDataset import load_tokenizer, BPEDataset



class AttentionNet(nn.Module):
  def __init__(self, num_heads: int, dim: int, oupt_dim: int, bias: bool = False):
    super(AttentionNet, self).__init__()
    self.num_heads, self.dim, self.bias, self.gamma = num_heads, dim, bias, 0.1

    self.at1 = MultiHeadAttention(dim=self.dim, num_heads=self.num_heads, bias=self.bias)
    self.fc1 = nn.Linear(self.dim, oupt_dim, bias=self.bias)
    self.swish, self.softmax = nn.SiLU(), nn.Softmax(dim=-1)
    self.norm, self.dropout = nn.LayerNorm(dim), nn.Dropout(p=self.gamma)
  # __init__()

  def forward(self, x):
    x = x + self.dropout(self.at1(self.norm(x)))
    x = self.swish(x)
    x = self.fc1(x)
    return self.swish(x)
  # forward()
# AttentionNet

class MultiHeadAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, bias: bool):
    super(MultiHeadAttention, self).__init__()
    assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
    self.dim, self.num_heads, self.sqrt_d_k = dim, num_heads, (dim // num_heads)**0.5

    self.w_q, self.w_k = nn.Linear(self.dim, self.dim, bias=bias), nn.Linear(self.dim, self.dim, bias=bias)
    self.w_v, self.w_o = nn.Linear(self.dim, self.dim, bias=bias), nn.Linear(self.dim, self.dim, bias=bias)
  # __init__()

  def forward(self, x):
    Q, K, V = self.w_q(x), self.w_k(x), self.w_v(x)
    raw_attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    down_scaled_raw_attn_scores = raw_attn_scores / self.sqrt_d_k
    attn_scores = F.softmax(down_scaled_raw_attn_scores, dim=-1)
    return torch.matmul(attn_scores, V)
  # attn_score()
# MultiHeadAttention

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
      nn.init.zeros_(m.bias)
# init_weights()

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
  model = AttentionNet(num_heads=CONFIG["num_heads"], dim=CONFIG["dim"], bias=False, oupt_dim=len(trainset.num_classes))
  model.to(device)
  model.apply(init_weights)
  nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

  # Training loop
  progress = tqdm(range(CONFIG["iterations"]), desc="training", unit="epoch", leave=True, dynamic_ncols=True)
  for _ in progress:
    for feature, label in DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      output = model.forward(feature)
      loss = criterion(output, label)
      optim.zero_grad()
      loss.backward()
      optim.step()
# if __name__ == "__main__":