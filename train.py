#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('corpus/Psalms_test.txt', new_model=True, num_epochs=10, gen_epochs=1, train_size=3, dropout=0.2, rnn_layers=3,   rnn_size=256,rnn_bidirectional=True, max_length=100, max_words=10000, dim_embeddings=100, word_level=False, single_text=False, name='Psalms')
textgen.generate()
print(textgen.model.summary())