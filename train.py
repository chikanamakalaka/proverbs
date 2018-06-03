#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('corpus/prov.txt', new_model=True, num_epochs=10, gen_epochs=5, train_size=3, dropout=0.2)
textgen.generate()