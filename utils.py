import io
import os
import unicodedata
import string
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

class helper:
    def __init__(self):
        self.ALL_LETTERS = string.ascii_letters + " .,;'"

    def unicode_to_ascii(self, s):
        return ''.join(c for c in s if unicodedata.category(c) != 'Mn' and c in self.ALL_LETTERS)

    def find_files(self, path):
        return glob.glob(path)

    def read_lines(self, filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicode_to_ascii(line) for line in lines]

    def random_choice(self, a):
        return a[random.randint(0, len(a) - 1)]
