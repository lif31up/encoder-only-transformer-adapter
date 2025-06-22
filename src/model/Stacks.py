import torch
from torch import nn

class EncoderStack(nn.Module):
  def __init__(self, config, init_weights=None):
    super(EncoderStack, self).__init__()
    self.at = MultiHeadAttention(config, init_weights=init_weights, mode="scaled")
    self.ffn = nn.ModuleList()
    for _ in range(config["n_hidn"]):
      self.ffn.append(nn.Linear(config["dim"], config["dim"], bias=config["bias"]))
    self.activation, self.ln = nn.GELU(), nn.LayerNorm(config["dim"])
    self.dropout = nn.Dropout(config["dropout"])

    if init_weights: self.ffn.apply(init_weights)
  # __init__()

  def forward(self, x):
    res = x
    x = self.ln(self.at(x) + res)
    res = x
    for i, fc in enumerate(self.ffn):
      if i != len(self.ffn): x = self.dropout(self.activation(fc(x)))
      else: x = self.dropout(fc(x))
    return self.ln(x + res)
  # forward(): it forwar-pass given input through all layers to produce output.
# EncoderStack

class DecoderStack(nn.Module):
  def __init__(self, config, init_weights=None):
    super(DecoderStack, self).__init__()
    self.masked_at = MultiHeadAttention(config, init_weights=init_weights, mode="scaled")
    self.cross_at = MultiHeadAttention(config, init_weights=init_weights, mode="cross")
    self.ffn = nn.ModuleList()
    for _ in range(config["n_hidn"]):
      self.ffn.append(nn.Linear(config["dim"], config["dim"], bias=config["bias"]))
    self.activation, self.ln = nn.GELU(), nn.LayerNorm(config["dim"])
    self.dropout = nn.Dropout(config["output"])

    if init_weights: self.ffn.apply(init_weights)
  # __init__()

  def forward(self, x, y):
    res = x
    x = self.ln(self.masked_at(x) + res)
    res = x
    x = self.ln(self.cross_at(x, y) + res)
    res = x
    for i, fc in enumerate(self.ffn):
      if i != len(self.ffn):
        x = self.dropout(self.activation(fc(x)))
      else:
        x = self.dropout(fc(x))
    return self.ln(x + res)
  # forward()
# decoder_stack

class MultiHeadAttention(nn.Module):
  def __init__(self, config, mode="scaled", init_weights=None):
    super(MultiHeadAttention, self).__init__()
    assert config["dim"] % config["num_heads"] == 0, "Dimension must be divisible by number of heads"
    self.sqrt_d_k, self.mode = (config["dim"] // config["num_heads"])**0.5, mode
    self.w_q, self.w_k = nn.Linear(config["dim"], config["dim"], bias=config["bias"]), nn.Linear(config["dim"], config["dim"], bias=config["bias"])
    self.w_v, self.w_o = nn.Linear(config["dim"], config["dim"], bias=config["bias"]), nn.Linear(config["dim"], config["dim"], bias=config["bias"])
    self.ln, self.dropout, self.softmax = nn.LayerNorm(config["dim"]), nn.Dropout(config["attention_dropout"]), nn.Softmax(dim=-1)

    if init_weights: self.apply(init_weights)
  # __init__()

  def forward(self, x, y=None):
    Q = self.w_q(x)
    (K, V) = (self.w_k(x), self.w_v(x)) if self.mode != "cross" else (self.w_k(y), self.w_v(y))
    raw_attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    down_scaled_raw_attn_scores = raw_attn_scores / self.sqrt_d_k
    attn_scores = self.softmax(down_scaled_raw_attn_scores)
    attn_scores = self.dropout(attn_scores)
    return self.ln(torch.matmul(attn_scores, V) + x)
  # attn_score()
# MultiHeadAttention