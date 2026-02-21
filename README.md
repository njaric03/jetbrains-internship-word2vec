## Word2Vec in NumPy

Pure NumPy implementation of the Skip-gram model with negative sampling (SGNS), including subsampling of frequent words, dynamic context windows, and linear learning rate decay. Trained on the [text8](http://mattmahoney.net/dc/text8.zip) dataset.

### Structure

```
main.py             training entry point
exploration.ipynb   embedding exploration and hyperparameter comparison
word2vec/
  data.py           tokenization, vocabulary, subsampling, training pairs
  model.py          embeddings, forward pass (loss + gradients), SGD update
  train.py          training loop, negative sampling, LR decay
```

### References I modeled the code after

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (Mikolov et al., 2013)
