import torch
from torch import nn
import torch.nn.functional as F

class EncoderStack(nn.Module):
  def __init__(self, dim, n_hidn, num_heads, bias=False):
    super(EncoderStack, self).__init__()
    self.dim, self.n_hidn, self.num_heads, self.bias = dim, n_hidn, num_heads, bias
    self.at = MultiHeadAttention(dim=self.dim, num_heads=self.num_heads, bias=self.bias)
    self.ffn = nn.ModuleList([nn.Linear(self.dim, self.dim, bias=self.bias) for _ in range(self.n_hidn)])
    for _ in range(self.n_hidn): self.ffn.append(nn.Linear(self.dim, self.dim, bias=self.bias))
    self.swish, self.ln = nn.SiLU(), nn.LayerNorm(self.dim)
  # __init__()

  def forward(self, input):
    input = self.at(input)
    residual = input
    for fc in self.ffn: input = self.swish(fc(input))
    return self.ln(self.swish(input) + residual)
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
