This implementation of the Multi-head Attention mechanism is based on the paper **"Attention is All You Need" by Vaswani et al. (2017)**.
- **task:** classifying movie reviews as positive or negative.
- **dataset:** `IMDB` dataset, which contains 50,000 movie reviews labeled as positive or negative.

# Attention-is-All-You-Need
Attention mechanisms are widely used in natural language processing tasks, particularly in transformer models.
The main goal of this implementation is to provide a clear and concise understanding of how multi-head attention works, including the key components such as query, key, value matrices, and the attention mechanism itself.

This implementation includes the following architectures:
* **BERT-like model** which uses only the encoder part of the transformer architecture.
* **GPT-like model** which uses only the decoder part of the transformer architecture.

> Check out the full explanation in [GitBook](https://lif31up.gitbook.io/lif31up/natural-language-process/attention-is-all-you-need)

> You can quickstart on [Colab](https://colab.research.google.com/drive/1IfCdclHqH4L0O1UlJrOViVncYQCNmaj1?usp=sharing)

### Instructions
`train.py`: This script is used to train the model on the IMDB dataset. It includes the training loop, evaluation, and saving the model checkpoints.
```python
if __name__ == "__main__":
  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  create_tokenizer(dataset["text"], "tokenizer.json")
  tokenizer, pretrained_model = load_tokenizer("tokenizer.json", PRETRAINED_MODEL=CONFIG["pretrained_model"])
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  vurt = BERT(num_heads=CONFIG["num_heads"], dim=CONFIG["dim"], oupt_dim=len(set(dataset["label"])), n_hidn=2, bias=False, n_stack=3).to(device)
  pity = GPT(num_heads=CONFIG["num_heads"], dim=CONFIG["dim"], oupt_dim=len(set(dataset["label"])), n_hidn=2, bias=False, n_stack=3).to(device)
  
  train(vurt, dataset, tokenizer, SAVE_TO="bert_model", clip_grad=True)
  train(pity, dataset, tokenizer, SAVE_TO="gpt_model", clip_grad=True)
```
`confing.py`: This file contains the configuration settings for the model, including the number of heads, dimensions, learning rate, and other hyperparameters.
```python
CONFIG = {
  "num_heads": 2,
  "dim": 2 * 32,  # num_heads * 8
  "iterations": 10,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "pretrained_model": "bert-base-uncased",
} # CONFIG
```
