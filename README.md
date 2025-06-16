This implementation of the Multi-head Attention mechanism is based on the paper **"Attention is All You Need" by Vaswani et al. (2017)**.
- **task:** classifying movie reviews as positive or negative.
- **dataset:** `IMDB` dataset, which contains 50,000 movie reviews labeled as positive or negative.

# Attention-is-All-You-Need
Attention mechanisms are widely used in natural language processing tasks, particularly in transformer models.
The main goal of this implementation is to provide a clear and concise understanding of how multi-head attention works, including the key components such as query, key, value matrices, and the attention mechanism itself.

> You can find the full documentation [here](https://lif31up.gitbook.io/lif31up/natural-language-process/attention-mechanism-the-core-of-modern-ai).

> You can access the test result on colab [here](https://colab.research.google.com/drive/1IfCdclHqH4L0O1UlJrOViVncYQCNmaj1?usp=sharing).

### More Explanation
Attention mechanisms allow models to focus on different parts of the input sequence when making predictions. The multi-head attention mechanism extends this idea by using multiple sets of query, key, and value matrices, allowing the model to capture different types of relationships in the data.
- **Query (Q):** Represents the current word or token we are focusing on.
- **Key (K):** Represents the words or tokens in the input sequence that we are comparing against.
- **Value (V):** Represents the actual information we want to retrieve based on the attention scores.
- **Attention Scores:** Calculated by taking the dot product of the query and key matrices, followed by a softmax operation to normalize the scores.