This implementation of the Multi-head Attention mechanism is based on the paper **"Attention is All You Need" by Vaswani et al. (2017)**.
- **task:** classifying movie reviews as positive or negative.
- **dataset:** `IMDB` dataset, which contains 50,000 movie reviews labeled as positive or negative.

# Attention-is-All-You-Need
Attention mechanisms are widely used in natural language processing tasks, particularly in transformer models.
The main goal of this implementation is to provide a clear and concise understanding of how multi-head attention works, including the key components such as query, key, value matrices, and the attention mechanism itself.

> You can find the full documentation [here](https://lif31up.gitbook.io/lif31up/natural-language-process/attention-mechanism-the-core-of-modern-ai).

> You can access the test result on colab [here](https://colab.research.google.com/drive/1IfCdclHqH4L0O1UlJrOViVncYQCNmaj1?usp=sharing).

### Implementation Overview
This implementation includes the following architectures:
* **BERT-like model** which uses only the encoder part of the transformer architecture.
* **GPT-like model** which uses only the decoder part of the transformer architecture.

### More Explanation
`src` and `src/model` directories contain the implementation of the transformer, including the following files:
- `BPEDataset.py`: This file contains the implementation of the BPE (Byte Pair Encoding) dataset class, which is used to preprocess the IMDB dataset.
- `BERT.py`: This file contains the implementation of the BERT-like model, which uses the encoder part of the transformer architecture.
- `GPT.py`: This file contains the implementation of the GPT-like model, which uses the decoder part of the transformer architecture.
- `Stacks.py`: This file contains the implementation of the stack class, which is used to stack the encoder and decoder layers.

### Instructions
You can run the code by `if __name__ == "__main__":` block in the `train.py` file. The main components of the code are as follows:
* `train.py`: This script is used to train the model on the IMDB dataset. It includes the training loop, evaluation, and saving the model checkpoints.
* `tokenizer.py`: This file contains the implementation of the tokenizer class, which is used to tokenize the input text.
* `config.py`: This file contains the configuration settings for the model, including hyperparameters such as learning rate, batch size, number of head and number of epochs.