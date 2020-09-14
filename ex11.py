import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

from ex9 import Spiral


def silverman_h(class_sd, N_in_class):
    return 1.06 * class_sd * (N_in_class ** 0.2)


def exp_kde(x, xi, h):
    upper = (x - xi) ** 2
    lower = 2 * h ** 2
    return np.prod(np.exp(-(upper) / lower))


def kde(x, data):
    N = data.shape[0]
    n = data.shape[1]
    h = silverman_h(np.std(data), N)
    mult = N * ((np.sqrt(2 * np.pi) * h) ** n)
    r = sum([exp_kde(x, d, h) for d in data])
    return r / mult


def main():
    spiral = Spiral('ex7.txt')
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(spiral.points):
        points_train = spiral.points[train_index]
        class_train = spiral.classification[train_index]

        points_train_1 = points_train[np.where(class_train == 1)]
        class_train_1 = class_train[np.where(class_train == 1)]

        points_train_2 = points_train[np.where(class_train == 2)]
        class_train_2 = class_train[np.where(class_train == 2)]

        points_test = spiral.points[test_index]
        class_test = spiral.classification[test_index]

        kde_1 = []
        ked_2 = []
        for p in points_test:
            kde_1.append(kde(p, points_train_1))
            ked_2.append(kde(p, points_train_2))

        norm.pdf(x_axis,0,2)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        print(silverman_h(np.std(points_train_1), points_train_1.shape[0]))


if __name__ == '__main__':
    main()
