import numpy as np

vocab = {}
vocab_size = 0
reviews = []

with open('reviews_limited_vocab.txt', 'r') as f:
    for line in f.readlines():
        words = line.strip().split(' ')
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_size
                vocab_size += 1
        reviews.append([vocab[word] for word in words])

invert_vocab = [''] * vocab_size
for (word, word_id) in vocab.items():
    invert_vocab[word_id] = word
invert_vocab = np.array(invert_vocab)

words_to_compare = ['excellent', 'amazing', 'delicious', 'fantastic', 'gem', 'perfectly', 'incredible', 'worst', 'mediocre', 'bland', 'meh', 'awful', 'horrible', 'terrible']

k_to_try = [ 2, 4, 8 ]

