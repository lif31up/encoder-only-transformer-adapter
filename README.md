This implementation of the Multi-head Attention mechanism is based on the paper **"Attention is All You Need" by Vaswani et al. (2017)**.
- **task:** classifying movie reviews as positive or negative.
- **dataset:** `IMDB` dataset, which contains 50,000 movie reviews labeled as positive or negative.
- **pretrained model:** `bert-base-uncased`, my implementation serves as late layers of pretrained model, which are used to fine-tune the model on the IMDB dataset.

# Attention-is-All-You-Need
Attention mechanisms are widely used in natural language processing tasks, particularly in transformer models.
The main goal of this implementation is to provide a clear and concise understanding of how multi-head attention works, including the key components such as query, key, value matrices, and the attention mechanism itself.

This implementation includes the following architectures:
* **BERT-like model** which uses only the encoder part of the transformer architecture.
* **GPT-like model** which uses only the decoder part of the transformer architecture.

> Check out the full explanation in [GitBook](https://lif31up.gitbook.io/lif31up/natural-language-process/attention-is-all-you-need)

> You can quickstart on [Colab](https://colab.research.google.com/drive/1oEwK7Tz-XvABJQ9-ypHznY24vD_uq4h_?usp=sharing)

> I referenced the following resources: [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)

> Download the model tensors on [Hugging Face](https://huggingface.co/lif31up/attention-is-all-you-need)

### Instructions
`confing.py`: This file contains the configuration settings for the model, including the number of heads, dimensions, learning rate, and other hyperparameters.
```python
CONFIG = {
  "version": "1.0",
  "model": {
    ...
  },  # model_config
  "tokenizer_config": {
    ...
  }, # tokenizer_config
  "iterations": 10,
  ...,
  "clip_grad": True,
} # CONFIG
```
`train.py`: This script is used to train the model on the IMDB dataset. It includes the training loop, evaluation, and saving the model checkpoints.
```python
if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  train(dataset, config=CONFIG, SAVE_TO="BERT")  # Replace with the actual model path
# __name__
```
`eval.py`: This script is used to evaluate the trained model on the IMDB dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.
```python
if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  evaluate("BERT.pth", dataset)  # Replace with the actual model path
# __name__
## output example: accuracy: 0.91
```

### Result
There are two types of results: first is the accuracy of the transferred model while second is the accuracy of the model trained from scratch.

|          | BERT(transferred)   | GPT(transferred)    |
|----------|---------------------|---------------------|
| **imdb** | `100%` **(100/100)** | `100%` **(100/100)** |
