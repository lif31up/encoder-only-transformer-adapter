from config import CONFIG
from src.BPE import create_tokenizer, embed
from src.model.AttentionNet import AttentionNet
from datasets import load_dataset

def train(SAVE_PATH: str):
  # overall configuration
  num_heads, dim, bias = CONFIG["num_heads"], CONFIG["dim"], CONFIG["bias"]
  model = AttentionNet(num_heads, dim, bias)

  dataset = load_dataset("imdb")
  tokenizer, model = create_tokenizer(dataset["train"]["text"], SAVE_PATH=SAVE_PATH)
  encode = lambda text: embed(text, tokenizer, model)

  
# train()