import numpy as np
from skimage import morphology
from collections import OrderedDict
from tensorflow.python.framework import ops


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x):
    for i in range(1, x.max() + 1):
        yield rle_encoding(x == i)
