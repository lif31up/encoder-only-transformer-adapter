from torch import nn
from src.model.Stacks import EncoderStack, DecoderStack, init_weights


class Transformer(nn.Module):
  def __init__(self, num_heads: int, dim: int, oupt_dim: int, n_hidn: int, bias: bool = False, n_stack: int = 3):
    super(Transformer, self).__init__()
    self.num_heads, self.bias, self.n_hidn = num_heads, bias, n_hidn
    self.dim, self.oupt_dim, self.n_stack = dim, oupt_dim, n_stack

    # define stacks
    self.encoder = nn.ModuleList([EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias) for _ in range(self.n_stack)])
    self.decoder = nn.ModuleList([DecoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias) for _ in range(self.n_stack)])

    # embedding, fully connected, and positional encoding layers
    self.fc = nn.Linear(self.dim, self.oupt_dim, bias=self.bias).apply(init_weights)
    self.softmax = nn.Softmax(dim=-1)
  # __init__()

  def forward(self, input, output):
    for encoder in self.encoder: input = encoder(input)
    for decoder in self.decoder: output = decoder(input, output)

    return self.softmax(self.fc(output))
  # forward()
# Transformer