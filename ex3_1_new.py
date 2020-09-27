import matplotlib.pyplot as plt
from numpy import ones, arange, array, sqrt, exp, cov, mean, concatenate, meshgrid, zeros
from numpy.linalg import det, inv
from numpy.random.mtrand import dirichlet, normal
from scipy.constants import pi
import numpy as np


def data_frontier(random_data, grid):
    x = grid
    y = grid

    m = zeros((len(x), len(y)))
    solver_k = [solver(grid, np.array(data).T) for data in random_data]
    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            values_list = [s[i][j] for s in solver_k]
            m[i][j] = values_list.index(max(values_list))
    return m


def solver(grid, data):
    x = grid
    y = grid

    m = zeros((len(x), len(y)))

    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            m[i][j] = pdf_2(2, cov(data.T), array([x_i, y_i]), mean(data, axis=0))
    return m


def data(N_attr, N, sd, mean, p):
    total = int(N * (1 - p)) + 1
    random_data = array([normal(size=total) * sd + mean[i] for i in range(N_attr)]).T
    training_data = array([normal(size=(N - total)) * sd + mean[i] for i in range(N_attr)]).T
    return training_data, random_data


def pdf(s1, s2, p, x, u1, y, u2):
    mul1_exp = -1 / (2 * (1 - p ** 2))
    mul2_exp = (x - u1) ** 2 / s1 ** 2 + (y - u2) ** 2 / s2 ** 2 - 2 * p * (x - u1) * (y - u2) / (s1 * s2)
    div = 2 * pi * s1 * s2 * sqrt(1 - p ** 2)
    return exp(mul1_exp * mul2_exp) / div


def pdf_2(n, K, x, m):
    d = 1 / sqrt(((2 * pi) ** n) * det(K))
    e = exp(-(0.5 * ((x - m) @ inv(K)) @ (x - m)))
    return d * e


def main():
    k1 = pdf(s1=1.1, s2=1, p=0, x=5.7, u1=2.5, y=4.2, u2=2.2)
    k2 = pdf(s1=0.8, s2=1.2, p=0, x=5.7, u1=6.1, y=4.2, u2=6.7)
    n_1 = 97 / (97 + 78)
    n_2 = 1 - n_1
    print((k1 * n_1) / (k2 * n_2))
    exit()
    training_percentage = 1.8
    x = dirichlet(ones(2), 1)[0] * training_percentage
    grid = arange(0.06, 6, 0.06)

    N = 200
    sds = [0.8, 0.4]
    means = [[2, 2], [4, 4]]

    c_1_training, c_1_random = data(2, N, sds[0], means[0], 0.9)
    c_2_training, c_2_random = data(2, N, sds[1], means[1], 0.9)
    data_con = concatenate((c_1_random, c_2_random))
    pdf_c1 = [
        pdf_2(2, cov(c_1_training.T), d, mean(c_1_training.T, axis=1)) for d in data_con
    ]
    pdf_c2 = [
        pdf_2(2, cov(c_2_training.T), d, mean(c_2_training.T, axis=1)) for d in data_con
    ]
    total = concatenate((c_1_training, c_2_training)).shape[0]
    n_c1 = c_1_training.shape[0] / total
    n_c2 = 1 - n_c1
    i = 0

    final_c1 = []
    final_c2 = []
    for x, y, d in zip(pdf_c1, pdf_c2, concatenate((c_1_random, c_2_random))):
        k = (x * n_c1) / (y * n_c2)
        if k < 1:
            final_c1.append(d)
        else:
            final_c2.append(d)
    final_c1 = array(final_c1)
    final_c2 = array(final_c2)
    print(f"Erros C1: {((final_c1.shape[0] - c_1_random.shape[0]) / c_1_random.shape[0]) * 100}%")
    print(f"Erros C2: {((final_c2.shape[0] - c_2_random.shape[0]) / c_2_random.shape[0]) * 100}%")
    scatter_plot([c_1_training.T, c_2_training.T]
                 , ['C1', 'C2'])
    scatter_plot([c_1_training.T, c_2_training.T, data_con.T]
                 , ['C1', 'C2', 'New Data'])
    scatter_plot([c_1_random.T, c_2_random.T]
                 , ['Should be C1', 'Should be C2'])
    scatter_plot([c_1_training.T, c_2_training.T, final_c1.T, final_c2.T]
                 , ['C1', 'C2', 'New C1', 'New C2'])

    c1_all = concatenate((c_1_training, final_c1)).T
    c2_all = concatenate((c_2_training, final_c2)).T
    frontier = {
        'grid_x': grid,
        'grid_y': grid,
        'frontier': data_frontier([c1_all, c2_all], grid)
    }

    scatter_plot([c1_all, c2_all]
                 , ['C1', 'C2'], frontier)


def scatter_plot(random_data, label, frontier=False):
    fig, ax = plt.subplots()
    for d, l in zip(random_data, label):
        ax.scatter(d[0], d[1], edgecolors='none', label=l)
        ax.grid(True)

    if frontier:
        X, Y = meshgrid(frontier['grid_x'], frontier['grid_y'])
        ax.contour(Y, X, frontier['frontier'])

    ax.legend()
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.show()


if __name__ == '__main__':
    main()
