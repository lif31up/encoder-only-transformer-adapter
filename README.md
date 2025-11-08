#  PETL Adaptor for Pretrained BERT
This implementation is inspired by:
[Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
[Parameter-Efficient Transfer Learning for NLP (2019)](https://arxiv.org/abs/1902.00751) by Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly.

In this implementation, I used an encoder-only transformer as the head for `bert-uncased-base`—the most common approach—to gain hands-on experience coding BERT from scratch.

- **Task:** Text Classification
- **Dataset:** IMDb Movie Reviews
- **Pretrained Model:** `bert-base-uncased`

### Requirements
To run the code on your own machine, `run pip install -r requirements.txt`.
```text
tqdm>=4.67.1
```

### Configuration
`confing.py` contains the configuration settings for the model, including the number of heads, dimensions, learning rate, and other hyperparameters.
```python
class Config: # free to tweak the params as you want
  def __init__(self):
    self.n_heads = 12
    self.n_stacks = 1
    self.n_hidden = 2
    self.dim = 768
    self.output_dim = 2
    self.bias = True

    self.dropout = 0.1
    self.attention_dropout = 0.1
    self.eps = 1e-3
    self.betas = (0.9, 0.98)
    self.epochs = 5
    self.batch_size = 16
    self.lr = 1e-4
    self.clip_grad = False
    self.mask_prob = 0.3

    self.pretrained_model = "bert-base-uncased"
    self.textset, self.testset_for_test = get_textset()
    self.save_to = "your_path"
    self.embedder, self.tokenizer = get_embedder(self.pretrained_model) ) # you can change the tokenizer setting on `./tokenizer.json`.
    self.dummy = embed(text='hello, world', model=self.embedder, tokenizer=self.tokenizer)
```
### Training
`train.py` is a script to train the model on the IMDB dataset. It includes the training loop, evaluation, and saving the model checkpoints.

```python
if __name__ == "__main__":
  from config import Config
  from model.Transformer import Transformer
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  bert_config = Config()
  trainset = EmbeddedDataset(
    dataset=bert_config.textset, dim=bert_config.dim, tokenizer=bert_config.tokenizer, embedder=bert_config.embedder)
  trainset.consolidate()
  bert_config.dummy = trainset[0][0]
  model = Transformer(bert_config)
  train(model=model, path=bert_config.save_to, trainset=trainset, config=bert_config, device=device)
# if __name__ == "__main__":
```
### Evaluation
`eval.py` is used to evaluate the trained model on the IMDB dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.
```python
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  bert_config = Config()
  my_data = torch.load(
    '/content/drive/MyDrive/Colab Notebooks/BERT.bin', map_location=torch.device('cpu'), weights_only=False)
  my_model = Transformer(my_data['config'])
  my_model.load_state_dict(my_data["state"])
  testset = EmbeddedDataset(
    dataset=bert_config.testset_for_test, dim=bert_config.dim, tokenizer=bert_config.tokenizer, embedder=bert_config.embedder)
  testset.consolidate()
  evaluate(model=my_model, dataset=testset, device=device)
#if __name__ == "__main__":
```