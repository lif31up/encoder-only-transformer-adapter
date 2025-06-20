from torch import nn
from src.model.Stacks import EncoderStack, DecoderStack


class Transformer(nn.Module):
  def __init__(self, model_config, init_weights=None):
    super(Transformer, self).__init__()
    self.num_heads, self.bias, self.n_hidn = model_config["num_heads"], model_config["bias"], model_config["n_hidn"]
    self.dim, self.oupt_dim, self.n_stack = model_config["dim"], model_config["oupt_dim"], model_config["n_stack"]

    self.encoders = nn.ModuleList([EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias) for _ in range(self.n_stack)])
    self.decoders = nn.ModuleList([DecoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias) for _ in range(self.n_stack)])
    self.fc = nn.Linear(self.dim, self.oupt_dim, bias=self.bias).apply(init_weights)
  # __init__()

  def forward(self, input, output):
    for encoder in self.encoders: input = encoder(input)
    for decoder in self.decoders: output = decoder(input, output)

    return self.fc(input.mean(dim=1))
  # forward()
# Transformer