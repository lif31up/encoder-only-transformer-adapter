This implementation of BERT-like Model is based on the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (2017) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
- **Hugging Face:** [Hugging Face](https://huggingface.co/lif31up/attention-is-all-you-need)
- **Note & Reference:** [GitBook](https://lif31up.gitbook.io/lif31up/natural-language-process/attention-is-all-you-need), [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)

# Attention-is-All-You-Need
Attention mechanisms are widely used in natural language processing tasks, particularly in transformer models.
The main goal of this implementation is to provide a clear and concise understanding of how multi-head attention works, including the key components such as query, key, value matrices, and the attention mechanism itself.
- **Task:** classifying movie reviews as positive or negative.
- **Dataset:** `IMDB` dataset, which contains 50,000 movie reviews labeled as positive or negative.
- **Pretrained Model:** `bert-base-uncased`, my implementation serves as late layers of pretrained model, which are used to fine-tune the model on the IMDB dataset.
- **Quickstart on Colab:** [Colab](https://colab.research.google.com/drive/1oEwK7Tz-XvABJQ9-ypHznY24vD_uq4h_?usp=sharing)

### Encoder and Decoder Stacks

The **encoder** consists of six identical layers. Each layer contains two sublayers: a multi-head self-attention mechanism and a simple, fully connected feed-forward network. These sublayers are connected through residual connections and layer normalization.
1. **Multi-Head Attention Layer:** multi-head attention mechanism → residual connection → layer normalization
2. **Feed Forward Layer:** feed-forward network → residual connection → layer normalization

The **decoder** also consists of six identical layers, but with three sublayers each. Like the encoder, it uses residual connections and layer normalization. The decoder's attention layer is unique—it combines information from two sources: its own previous outputs through a *"masked multi-head self-attention mechanism"* and the encoder's outputs via an *"encoder/decoder attention"* layer.
1. **Masked Attention Layer:** masked self-attention → residual connection → layer normalization
2. **Cross Attention Layer:** multi-head attention mechanism → residual connection → layer normalization
3. **Feed Forward Layer:** feed-forward network → residual connection → layer normalization

The Transformer architecture has many variants, such as BERT-like and GPT-like models. BERT-like models use only the encoder, while GPT-like models use only the decoder, replacing the encoder output with the previous layer's output when calculating the cross-attention layer.

This implementation includes the following architectures:
* **BERT-like model** which uses only the encoder part of the transformer architecture. 
* ~~**GPT-like model** which uses only the decoder part of the transformer architecture.~~

### BERT-like Models

BERT (Bidirectional Encoder Representations from Transformers) and its variants represent a significant innovation in transformer architecture by utilizing only the encoder stack for bidirectional context understanding.

- **Bidirectional Context:** Unlike traditional left-to-right language models, BERT processes text bidirectionally, allowing it to understand context from both directions simultaneously.
- **Pre-training Tasks:** BERT uses two main pre-training objectives:
    - Masked Language Modeling (MLM): Randomly masks tokens and predicts them using bidirectional context
    - Next Sentence Prediction (NSP): Predicts whether two sentences naturally follow each other
- **Architecture Modifications:** BERT modifies the original transformer encoder by:
    - Removing the decoder stack entirely
    - Adding special tokens ([CLS], [SEP]) for specific tasks
    - Using learned positional embeddings instead of sinusoidal

Notable BERT variants include RoBERTa (modified pre-training), DistilBERT (compressed version), and ALBERT (parameter-efficient architecture). These models have achieved state-of-the-art results in various natural language understanding tasks.

- Text Classification
- Named Entity Recognition
- Question Answering
- Natural Language Inference

### About Transfer Learning
Transfer Learning is a technique to reuse already trained model for different tasks. This is

- In the pre-training phase, the model learns generic features from an enormous dataset.
- After pre-training, the model undergoes Transfer/Fine-Tuning through two possible approaches:
  - **Option 1: Retrain only the new later layers**
  - ~~Option 2: Retrain all layers~~

In this implementation, they will use **option 1 trafer** with `bert-base-uncased`. The above implements serve as the new later layers of the pretrained model.

---
### Configuration
`confing.py` contains the configuration settings for the model, including the number of heads, dimensions, learning rate, and other hyperparameters.
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
### Training
`train.py` is a script to train the model on the IMDB dataset. It includes the training loop, evaluation, and saving the model checkpoints.
```python
if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  train(dataset, config=CONFIG, SAVE_TO="BERT")  # Replace with the actual model path
# __name__
```
### Evaluation
`eval.py` is used to evaluate the trained model on the IMDB dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.
```python
if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  evaluate("BERT.pth", dataset)  # Replace with the actual model path
# __name__
## output example: accuracy: 0.91
```
---
### Result
There are two types of results: first is the accuracy of the transferred model while second is the accuracy of the model trained from scratch.

|          | BERT(transferred)   |
|----------|---------------------|
| **imdb** | `100%` **(100/100)** |
