This reimplementation of BERT-like Model with LoRA is based on the papers ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (2017) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin and ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (2021) by Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.

- **Hugging Face:** [Hugging Face](https://huggingface.co/lif31up/attention-is-all-you-need)
- **Note & Reference:** [GitBook](https://lif31up.gitbook.io/lif31up/natural-language-process/attention-is-all-you-need), [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)
- **Quickstart on Colab:** [Colab](https://colab.research.google.com/drive/1oEwK7Tz-XvABJQ9-ypHznY24vD_uq4h_?usp=sharing)

|          | BERT(transferred)   |
|----------|---------------------|
| **imdb** | `100%` **(500/500)** |

# An Encoder as a Low-Rank Adapter/Head for LLMs
This implementation is a variant of LoRA for `bert-base-uncased`. A newly extended encoder is placed as an adapter/head to learn down-streamed tasks.

- **Task:** classifying movie reviews as positive or negative.
- **Dataset:** `IMDB` dataset, which contains 50,000 movie reviews labeled as positive or negative.
- **Pretrained Model:** `bert-base-uncased`, my implementation serves as late layers of pretrained model, which are used to fine-tune the model on the IMDB dataset.

### Encoder Stacks
The **encoder** contains two sublayers: a multi-head self-attention mechanism and a simple, fully connected feed-forward network. These sublayers are connected through residual connections and layer normalization.
1. **Multi-Head Attention Layer:** multi-head attention mechanism → residual connection → layer normalization
2. **Feed Forward Layer:** feed-forward network → residual connection → layer normalization

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
## Technical Highlights
The BERT-like model architecture consists of several key components, including multi-head attention, feed-forward networks, and layer normalization. The model is designed to process input sequences and extract meaningful features for various natural language processing tasks.

### Multi-Head Attention
The model employs a multi-head attention mechanism, which allows it to focus on different parts of the input sequence simultaneously. This is achieved by projecting the input into multiple subspaces, computing attention for each subspace, and then concatenating the results.

```python
class MultiHeadAttention(nn.Module):
  def __init__(self, config, mode="scaled", init_weights=None):
    super(MultiHeadAttention, self).__init__()
    assert config["dim"] % config["num_heads"] == 0, "Dimension must be divisible by number of heads"
    self.config = config
    self.sqrt_d_k, self.mode = (config["dim"] // config["num_heads"])**0.5, mode
    self.w_q, self.w_k = nn.Linear(config["dim"], config["dim"], bias=config["bias"]), nn.Linear(config["dim"], config["dim"], bias=config["bias"])
    self.w_v, self.w_o = nn.Linear(config["dim"], config["dim"], bias=config["bias"]), nn.Linear(config["dim"], config["dim"], bias=config["bias"])
    self.ln, self.dropout, self.softmax = nn.LayerNorm(config["dim"]), nn.Dropout(config["attention_dropout"]), nn.Softmax(dim=-1)

    if init_weights: self.apply(init_weights)
  # __init__()

  def forward(self, x, y=None):
    Q = self.w_q(x)
    (K, V) = (self.w_k(x), self.w_v(x)) if self.mode != "cross" else (self.w_k(y), self.w_v(y))
    raw_attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    down_scaled_raw_attn_scores = raw_attn_scores / self.sqrt_d_k
    if self.mode == "masked":
      "Masking is not implemented in this example, but it would typically involve setting certain positions in the attention scores to a very low value (e.g., -inf) to prevent attention to those positions."
    attn_scores = self.softmax(down_scaled_raw_attn_scores)
    attn_scores = self.dropout(attn_scores)
    return self.ln(torch.matmul(attn_scores, V) + x)
  # attn_score()
# MultiHeadAttention
```
### Encoder Stack
The stack consists of multiple layers of multi-head attention and feed-forward networks. Each layer applies a multi-head attention mechanism followed by a feed-forward network, with residual connections and layer normalization applied at each step.
* Since the model is BERT-like and text classification task that does not require unidirectional attention(masked attention), the `mode` is set to `"scaled"` for the multi-head attention mechanism. However, masking is performed at the input layer by `BPEDataset`. This is common convention for modern BERT-like models.
* The feed-forward network consists of multiple linear layers with GELU activation functions, allowing the model to learn complex representations of the input data.
* Modern BERT implementations often locate the layer normalization firstly, which is different from the original paper. This implementation follows original convention.
```python
class EncoderStack(nn.Module):
  def __init__(self, config, init_weights=None):
    super(EncoderStack, self).__init__()
    self.at = MultiHeadAttention(config, init_weights=init_weights, mode="scaled")
    self.ffn = nn.ModuleList()
    for _ in range(config["n_hidn"]):
      self.ffn.append(nn.Linear(config["dim"], config["dim"], bias=config["bias"]))
    self.activation, self.ln = nn.GELU(), nn.LayerNorm(config["dim"])
    self.dropout = nn.Dropout(config["dropout"])

    if init_weights: self.ffn.apply(init_weights)
  # __init__()

  def forward(self, x):
    res = x
    x = self.ln(self.at(x) + res)
    res = x
    for i, fc in enumerate(self.ffn):
      if i != len(self.ffn): x = self.dropout(self.activation(fc(x)))
      else: x = self.dropout(fc(x))
    return self.ln(x + res)
  # forward(): it forwar-pass given input through all layers to produce output.
# EncoderStack
```
### Model
The BERT model is constructed using multiple encoder stacks. Each stack processes the input sequentially, applying multi-head attention and feed-forward networks to extract features from the input data.
```python
class BERT(nn.Module):
  def __init__(self, config, init_weights=None):
    super(BERT, self).__init__()
    self.stacks = nn.ModuleList([EncoderStack(config, init_weights=init_weights) for _ in range(config["n_stack"])])
    self.fc, self.flatten = nn.Linear(393216, config["oupt_dim"], bias=config["bias"]), nn.Flatten(1)
  # __init__():

  def forward(self, x):
    for stack in self.stacks: x = stack(x)
    return self.fc(self.flatten(x))
  # forward()
# BERT
```
