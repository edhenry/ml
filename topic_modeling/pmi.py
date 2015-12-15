__author__ = 'edhenry'


import os
import sys
import csv
import copy
import random
import itertools
from operator import itemgetter
from collections import defaultdict

# Make sure you've got Numpy and Scipy installed:
import numpy as np
import scipy
import scipy.spatial.distance
from numpy.linalg import svd

# For visualization:
from tsne import tsne # See http://lvdmaaten.github.io/tsne/#implementations
import matplotlib.pyplot as plt

# For clustering in the 'Word-sense ambiguities' section:
from sklearn.cluster import AffinityPropagation




