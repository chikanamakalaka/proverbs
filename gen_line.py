#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
from textgenrnn import textgenrnn

textgen = textgenrnn('textgenrnn_weights.hdf5')
textgen.generate(10, temperature=0.75)
