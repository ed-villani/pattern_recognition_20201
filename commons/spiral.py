from copy import deepcopy
import numpy as np


class Spiral:
    def __init__(self, path='/Users/eduardovillani/git/pattern_recognition_20201/data/spiral.txt'):
        spiral = []
        with open(path) as f:
            next(f)
            for line in f.readlines():
                k = line.replace('\n', '').split(',')
                k[0] = int(k[0])
                k[1] = float(k[1])
                k[2] = float(k[2])
                k[3] = int(k[3])
                spiral.append(k)
        del k
        del line
        del f
        self._spiral = np.array(spiral)

    @property
    def data(self):
        return deepcopy(self._spiral)

    @property
    def points(self):
        return deepcopy(self._spiral[:, 1:-1])

    @property
    def classification(self):
        return deepcopy(self._spiral[:, -1] - 1)
