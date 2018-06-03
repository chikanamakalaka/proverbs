#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('corpus/prov.txt', num_epochs=10)
textgen.generate()