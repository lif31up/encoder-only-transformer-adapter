from src.model.Stacks import DecoderStack
from torch import nn

class GPT(nn.Module):
  def __init__(self, num_heads: int, dim: int, oupt_dim: int, n_hidn: int, bias: bool = False, n_stack: int = 3):
    super(GPT, self).__init__()
    self.num_heads, self.bias, self.n_hidn = num_heads, bias, n_hidn
    self.dim, self.oupt_dim, self.n_stack = dim, oupt_dim, n_stack

    # define stacks
    self.decoder = nn.ModuleList([DecoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias) for _ in range(self.n_stack)])

    # embedding and fully connected layers
    self.fc = nn.Linear(self.dim, self.oupt_dim, bias=self.bias)
    self.softmax = nn.Softmax(dim=-1)
  # __init__()

  def forward(self, input):
    for decoder in self.decoder: input = decoder(input, input)
    return self.softmax(self.fc(input))
  # forward()
# BERT