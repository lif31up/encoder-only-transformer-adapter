from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def create_tokenizer(DATASET, SAVE_PATH: str):
  # init tokenizer
  tokenizer = Tokenizer(models.BPE())
  trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[PAD]", "[CLS]"])
  tokenizer.train_from_iterator(DATASET, trainer)  # Train on IMDb text

  # save tokenizer
  tokenizer.save(SAVE_PATH)
# init_tokenizer(): Initializes a Byte-Pair Encoding (BPE) tokenizer with a specified vocabulary size and special tokens.

if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset("imdb")
  create_tokenizer(dataset["train"]["text"], SAVE_PATH="../src/tokenizer.json")
  print("Tokenizer trained and saved to tokenizer.json")
# __name__