#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('Psa.txt', new_model=False, num_epochs=5, gen_epochs=1, train_size=3, dropout=0.2)
textgen.generate()
print(textgen.model.summary())
