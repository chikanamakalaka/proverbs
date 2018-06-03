#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
from textgenrnn import textgenrnn
import argparse
import sys

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Generates Lines')
	parser.add_argument('a', help = "How many lines to generate.")

args = parser.parse_args(sys.argv[1])
textgen = textgenrnn('weights/proverbs_weights.hdf5')
textgen.generate(int(sys.argv[1]), temperature=0.5)