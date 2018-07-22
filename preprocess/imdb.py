import os
import PIL
import numpy as np
import scipy.sparse

class imdb(object):
    def __init__(self, name):
        self.name = name
        self.num_classes = 0
        self.classes = []
        self.image_index = []

