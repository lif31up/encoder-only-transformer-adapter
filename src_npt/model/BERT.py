from torch import nn
from transformers import BertTokenizer, BertModel
from src.model.Stacks import EncoderStack

class BERT(nn.Module):
  def __init__(self, model_config, init_weights=None):
    super(BERT, self).__init__()
    self.num_heads, self.bias, self.n_hidn = model_config["num_heads"], model_config["bias"], model_config["n_hidn"]
    self.dim, self.oupt_dim, self.n_stack = model_config["dim"], model_config["oupt_dim"], model_config["n_stack"]

    self.stacks = nn.ModuleList([EncoderStack(dim=self.dim, n_hidn=self.n_hidn, num_heads=self.num_heads, bias=self.bias, init_weights=init_weights) for _ in range(self.n_stack)])
    self.fc = nn.Linear(self.dim, self.oupt_dim, bias=self.bias)
    self.softmax = nn.Softmax(dim=-1)
  # __init__():

  def forward(self, input):
    for stack in self.stacks: input = stack(input)
    return self.fc(input)
  # forward()
# BERT

if __name__ == "__main__":
  from datasets import load_dataset
  from torch.utils.data import DataLoader
  from src_npt.config import CONFIG
  from src_npt.BPEDataset import BPEDataset

  tokenizer_config, model_config = CONFIG["tokenizer_config"], CONFIG["model"]
  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bert_tokenizer.truncation_side = "right"
  bert_tokenizer.padding_side = "right"
  bert_tokenizer.pad_token = bert_tokenizer.eos_token = "[PAD]"
  bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
  trainset = BPEDataset(dataset=dataset, dim=model_config["dim"], tokenizer=bert_tokenizer, model=bert_model)

  def init_weights(m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None: nn.init.zeros_(m.bias)
  # init_weights()
  vurt = BERT(model_config, init_weights=None)

  for feature, label in DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4):
    output = vurt(feature)
    print(f"Feature shape: {feature.shape}, Label shape: {label.shape}")
    print(f"Output shape: {output.shape}")
    break  # Just to check the first batch
# if __name__ == "__main__":