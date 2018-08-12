import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import imdb

imdb.maybe_download_and_extract()
x_train_text, y_train = imdb.load_data(train=True)
x_test_text, y_test = imdb.load_data(train=False)

data_text = x_test_text+x_test_text

print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))

print("Train sample ", x_test_text[1])
print("Review result ", y_train[1])

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(data_text)

if num_words is None:
    num_words = len(tokenizer.word_index)

x_train_tokens = tokenizer.texts_to_sequences(x_train_text)

'''
print('Map ' , x_train_text[1])
print('Equivlant to ', x_train_tokens[1])
'''

x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)


max_tokensmax_toke  = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print('Number of max tokens', max_tokens)
print('Coverage')
print(np.sum(num_tokens < max_tokens) / len(num_tokens))

pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)

x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)


idx = tokenizertokenize .word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]
    
    # Concatenate all words.
    text = " ".join(words)

    return text


model = Sequential()
embedding_size = 8

model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))

model.add(GRU(units=16, return_sequences=True))

model.add(GRU(units=8, return_sequences=True))

model.add(GRU(units=4))

