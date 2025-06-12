import torch.optim
from torch import nn
import torch.nn.functional as F

class AttentionNet(nn.Module):
  def __init__(self, num_heads: int, dim: int, bias: bool):
    super(AttentionNet, self).__init__()
    self.num_heads, self.dim, self.bias = num_heads, dim, bias

    self.at1 = MultiHeadAttention(dim=self.dim, num_heads=8, bias=self.bias)
    self.fc1 = nn.Linear(self.dim, self.dim, bias=self.bias)
    self.at2 = MultiHeadAttention(dim=self.dim, num_heads=8, bias=self.bias)
    self.fc2 = nn.Linear(self.dim, self.dim, bias=self.bias)
    self.swish, self.softmax = nn.SiLU(), nn.Softmax(dim=-1)
  # __init__()

  def forward(self, x):
    x = self.at1(x)
    x = self.swish(x)
    x = self.fc1(x)
    x = self.at2(x)
    x = self.swish(x)
    x = self.fc2(x)
    return self.softmax(x)
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

if __name__ == "__main__":
  from config import CONFIG

  model = AttentionNet(num_heads=CONFIG["num_heads"], dim=CONFIG["dim"], bias=CONFIG["bias"])
  x = torch.randn(1, 10, CONFIG["dim"])
  output = model(x)
  print(output.shape)
# if __name__ == "__main__":