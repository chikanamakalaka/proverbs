#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('corpus/lorem.txt', num_epochs=10, gen_epochs=1, train_size=3,  rnn_bidirectional=True, max_length=100, max_words=10000, dim_embeddings=100, word_level=False, single_text=False, name='Lorem')
# textgen.train_from_file('corpus/Proverbs.txt', num_epochs=30, gen_epochs=1, name='Proverbs', dropout=0.5, rnn_layers=3, rnn_size=256, single_text=False)
print(textgen.model.summary())
