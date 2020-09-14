from copy import deepcopy
from math import sqrt
import numpy as np
from numpy import ones, cov, concatenate, vstack, corrcoef, pi
from numpy.linalg import det
from numpy.ma import arange, exp
from numpy.random.mtrand import dirichlet
import matplotlib.pyplot as plt
from random_data import RandomData, pdf, data_frontier


def scatter_plot(random_data, frontier=False):
    fig, ax = plt.subplots()
    for d in random_data:
        ax.scatter(d[0], d[1], alpha=0.3, edgecolors='none')
        ax.legend()
        ax.grid(True)
    if frontier:
        X, Y = np.meshgrid(frontier['grid_x'], frontier['grid_y'])
        ax.contour(Y, X, frontier['frontier'])
    plt.show()


def classifier(data, classes):
    all_data = deepcopy(data)
    all_data = concatenate(tuple(d for d in all_data), 1).T

    def pdf_k(n, K, x, m):
        d = 1 / sqrt((2 * pi) ** n * det(K))
        e = exp(0.5 * (x - m).T * K ** -1 * (x - m))
        return d * e

    # pc = probability_per_class(data)
    for d in all_data:
        pdfs = [
            pdf(
                s1=c.attr_sd_training[0],
                s2=c.attr_sd_training[1],
                p=corrcoef(c.traning_data)[0][1],
                u1=c.attr_mean_training[0],
                x=d[0],
                y=d[1],
                u2=c.attr_mean_training[1]
            ) for c in classes
        ]

        classes[pdfs.index(max(pdfs))].traning_data = vstack([classes[pdfs.index(max(pdfs))].traning_data.T, d]).T
        classes[pdfs.index(max(pdfs))].classifier += 1
    return


def main():
    training_percentage = 0.9
    x = dirichlet(ones(2), 1)[0] * training_percentage
    grid = arange(0.06, 6, 0.06)
    N = 200
    sds = [0.8, 0.4]
    means = [[2, 2], [4, 4]]
    data = [RandomData(N, sd, mean, arange(0.06, 6, 0.06), split=p) for sd, mean, p in zip(sds, means, x)]
    training = [d.traning_data for d in data]
    # scatter_plot(training)
    training.append(concatenate(tuple(d.random_data for d in data), 1))
    # scatter_plot(training)
    classifier([d.random_data for d in data], data)
    for d in data:
        d.random_data = d.traning_data
    frontier = {
        'grid_x': grid,
        'grid_y': grid,
        'frontier': data_frontier(data, grid)
    }
    scatter_plot([d.random_data for d in data], frontier)


def probability_per_class(data):
    p_c = []
    total = sum([data.shape[1] for data in data])
    for data in data:
        p_c.append((data.shape[1] / total))
    return p_c


if __name__ == '__main__':
    main()
