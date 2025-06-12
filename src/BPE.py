import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import AutoModel

def create_tokenizer(DATASET, SAVE_PATH: str):
  # init tokenizer
  tokenizer = Tokenizer(models.BPE())
  trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[PAD]", "[CLS]"])
  tokenizer.train_from_iterator(DATASET, trainer)  # Train on IMDb text

  # save tokenizer
  tokenizer.save(SAVE_PATH)
  return tokenizer, AutoModel.from_pretrained("bert-base-uncased")
# init_tokenizer(): Initializes a Byte-Pair Encoding (BPE) tokenizer with a specified vocabulary size and special tokens.

def embed(text: str, tokenizer: Tokenizer, model: AutoModel):
  input_ids = torch.tensor([tokenizer.encode(text).ids])
  with torch.no_grad(): outputs = model(input_ids)
  return torch.mean(outputs.last_hidden_state, dim=1)
# embed

save_path, saving = "tokenizer.json", False  # Path to save the tokenizer
if __name__ == "__main__":
  if saving:
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    create_tokenizer(dataset["train"]["text"], SAVE_PATH=save_path)
    print(f"Tokenizer created and saved to {save_path}.")
  else:
    tokenizer = Tokenizer.from_file("tokenizer.json")
    model = AutoModel.from_pretrained("bert-base-uncased")
    print(embed("your text here", tokenizer, model).shape)
# __name__