#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('corpus/Psalms.txt', new_model=True, num_epochs=50, gen_epochs=10, train_size=2, rnn_layers=3, rnn_size=256, rnn_bidirectional=False, max_length=100, max_words=10000, dim_embeddings=100, word_level=False, single_text=False, name='Proverbs')
# textgen.train_from_file('corpus/Proverbs.txt', num_epochs=30, gen_epochs=1, name='Proverbs', dropout=0.5, single_text=False)
print(textgen.model.summary())