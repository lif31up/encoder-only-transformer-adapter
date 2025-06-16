from datasets import load_dataset
import torch.optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.BPEDataset import load_tokenizer, BPEDataset

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
# AttentionNet

class EncoderStack(nn.Module):
  def __init__(self, dim, n_hidn, num_heads, bias=False):
    super(EncoderStack, self).__init__()
    self.dim, self.n_hidn, self.num_heads, self.bias = dim, n_hidn, num_heads, bias
    self.at = MultiHeadAttention(dim=self.dim, num_heads=self.num_heads, bias=self.bias)
    self.ffn, self.swish, self.ln = nn.Linear(self.dim, self.n_hidn), nn.SiLU(), nn.LayerNorm(self.dim)
  # __init__()

  def forward(self, input):
    input = self.at(input)
    return self.ln(self.swish(self.ffn(input)) + input)
  # forward()
# EncoderStack

class DecoderStack(nn.Module):
  def __init__(self, dim, n_hidn, num_heads, bias=False):
    super(DecoderStack, self).__init__()
    self.dim, self.n_hidn, self.num_heads, self.bias = dim, n_hidn, num_heads, bias
    self.at1 = MultiHeadAttention(dim=self.dim, num_heads=self.num_heads, bias=self.bias)
    self.at2 = MultiHeadAttention(dim=self.dim, num_heads=self.num_heads, bias=self.bias, mode="cross")
    self.ffn, self.swish, self.ln = nn.Linear(self.dim, self.n_hidn), nn.SiLU(), nn.LayerNorm(self.dim)
  # __init__()

  def forward(self, input, output):
    input = self.at1(input, output)
    input = self.at2(input, output)
    return self.ln(self.swish(self.ffn(input)) + input)
  # forward()
# decoder_stack

class MultiHeadAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, bias: bool, mode="scaled"):
    super(MultiHeadAttention, self).__init__()
    assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
    self.dim, self.num_heads, self.sqrt_d_k, self.mode = dim, num_heads, (dim // num_heads)**0.5, mode

    self.w_q, self.w_k = nn.Linear(self.dim, self.dim, bias=bias), nn.Linear(self.dim, self.dim, bias=bias)
    self.w_v, self.w_o = nn.Linear(self.dim, self.dim, bias=bias), nn.Linear(self.dim, self.dim, bias=bias)
  # __init__()

  def forward(self, input, output=None):
    Q = self.w_q(input)
    K, V = self.w_k(input), self.w_v(input) if self.mode != "cross" else (self.w_k(output), self.w_v(output))
    raw_attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    down_scaled_raw_attn_scores = raw_attn_scores / self.sqrt_d_k
    attn_scores = F.softmax(down_scaled_raw_attn_scores, dim=-1)
    return F.layer_norm(torch.matmul(attn_scores, V) + input, normalized_shape=(self.dim,))
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