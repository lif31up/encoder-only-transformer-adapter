import torch
from holoviews.examples.reference.elements.bokeh.Div import df_html
from torch import nn


class Transformer(nn.Module):
  def __init__(self, config):
    super(Transformer, self).__init__()
    self.stacks = nn.ModuleList([EncoderStack(config) for _ in range(config.n_stacks)])
    self.fc, self.flatten = self._get_fc(self.config.dummy), nn.Flatten(start_dim=1)
  # __init__

  def forward(self, x):
    for stack in self.stacks: x = stack(x)
    return self.fc(self.flatten(x)).apply(self.config.init_weights)
  # forward

  def _get_fc(self, dummy):
    with torch.no_grad():
      for stack in self.stacks: dummy = stack(dummy)
    dummy = self.flatten(dummy)
    return nn.Linear(dummy.shape[1], self.config.output_dim, bias=self.config.bias)
  # _get_fc
# BERT

class Distill_Transformer(Transformer):
  def __init__(self, config):
    super(Distill_Transformer, self).__init__(config)
    assert self.config.distill, "self.config.distill is not True."
    self.alpha = self.config.alpha
  #__init__
# Distill_Transformer

class EncoderStack(nn.Module):
  def __init__(self, config):
    super(EncoderStack, self).__init__()
    self.mt_attn = MultiHeadAttention(config, mode="scaled")
    self.ffn = nn.ModuleList()
    for _ in range(config.n_hidden):
      self.ffn.append(nn.Linear(config.dim, config.dim, bias=config.bias))
    self.activation, self.ln = nn.GELU(), nn.LayerNorm(config.dim)
    self.dropout = nn.Dropout(config.dropout)

    self.apply(self.config.init_weights)
  # __init__

  def forward(self, x):
    res = x
    x = self.ln(self.mt_attn(x) + res)
    res = x
    for i, layer in enumerate(self.ffn):
      if i != len(self.ffn): x = self.dropout(self.activation(layer(x)))
      else: x = self.dropout(layer(x))
    return self.ln(x + res)
  # forward
# EncoderStack

class MultiHeadAttention(nn.Module):
  def __init__(self, config, mode="scaled"):
    super(MultiHeadAttention, self).__init__()
    assert config.dim % config.n_heads == 0, "Dimension must be divisible by number of heads"
    self.config = config
    self.sqrt_d_k, self.mode = (config.dim // config.n_heads) ** 0.5, mode
    self.w_q, self.w_k = nn.Linear(config.dim, config.dim, bias=config.bias), nn.Linear(config.dim, config.dim, bias=config.bias)
    self.w_v, self.w_o = nn.Linear(config.dim, config.dim, bias=config.bias), nn.Linear(config.dim, config.dim, bias=config.bias)
    self.ln, self.dropout, self.softmax = nn.LayerNorm(config.dim), nn.Dropout(config.attention_dropout), nn.Softmax(dim=1)

    self.apply(self.config.init_weights)
  # __init__

  def forward(self, x, y=None):
    Q = self.w_q(x)
    (K, V) = (self.w_k(x), self.w_v(x)) if self.mode != "cross" else (self.w_k(y), self.w_v(y))
    raw_attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    down_scaled_raw_attn_scores = raw_attn_scores / self.sqrt_d_k
    if self.mode == "masked":
      masked_indices = torch.rand(*down_scaled_raw_attn_scores.shape[:-1], 1) < self.config.mask_prob
      down_scaled_raw_attn_scores[masked_indices] = float("-inf")
    attn_scores = self.softmax(down_scaled_raw_attn_scores)
    attn_scores = self.dropout(attn_scores)
    return self.ln(torch.matmul(attn_scores, V) + x)
  # attn_score
# MultiHeadAttention