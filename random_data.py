from math import sqrt, pi, exp

import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, std, zeros, array
from numpy.random.mtrand import normal


class RandomData:
    def __init__(self, n, s, data_mean, grid, split=0):
        self._n = n
        self._s = s
        self._mean = data_mean
        self.grid_x = grid
        self.grid_y = grid

        # total = int(self._n * (1 - split)) + 1
        total = self._n
        self.random_data = array([normal(size=total) * self._s + self._mean[i] for i in range(2)])
        # self.traning_data = array([normal(size=(self._n - total)) * self._s + self._mean[i] for i in range(2)])
        self.classifier = 0

    @property
    def solver(self):
        return self.solver()

    @property
    def attr_mean_training(self):
        return [mean(attr) for attr in self.traning_data]

    @property
    def attr_sd_training(self):
        return [std(attr) for attr in self.traning_data]

    @property
    def attr_mean(self):
        return [mean(attr) for attr in self.random_data]

    @property
    def attr_sd(self):
        return [std(attr) for attr in self.random_data]

    def solver(self):
        x = self.grid_x
        y = self.grid_y

        m = zeros((len(x), len(y)))

        for i, x_i in enumerate(x):
            for j, y_i in enumerate(y):
                m[i][j] = pdf(
                    x=x_i,
                    y=y_i,
                    u1=self.attr_mean[0],
                    u2=self.attr_mean[1],
                    s1=self.attr_sd[0],
                    s2=self.attr_sd[1],
                    p=0
                )
        return m


def data_frontier(random_data, grid):
    x = grid
    y = grid

    m = zeros((len(x), len(y)))
    solver_k = [data.solver() for data in random_data]
    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            values_list = [s[i][j] for s in solver_k]
            m[i][j] = values_list.index(max(values_list))
    return m


class RandomDataPlotter:
    @staticmethod
    def surface_plot(random_data):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for data in random_data:
            X, Y = np.meshgrid(data.grid_x, data.grid_y)
            CS = ax.plot_surface(
                X,
                Y,
                data.solver(),
                rstride=1,
                cstride=1,
                cmap='viridis',
                edgecolor='none'
            )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_zlim(0, 0.4)
        fig.colorbar(CS, shrink=0.8, extend='both')
        plt.show()

    @staticmethod
    def scatter_plot(random_data, frontier=False):
        fig, ax = plt.subplots()
        for d in random_data:
            ax.scatter(d.random_data[0], d.random_data[1], alpha=0.3, edgecolors='none')
            ax.legend()
            ax.grid(True)
        if frontier:
            X, Y = np.meshgrid(frontier['grid_x'], frontier['grid_y'])
            ax.contour(Y, X, frontier['frontier'])
        plt.show()

    @staticmethod
    def contour_plot(random_data):
        fig, ax = plt.subplots()
        for data in random_data:
            X, Y = np.meshgrid(data.grid_x, data.grid_y)
            CS = ax.contour(Y, X, data.solver)
        fig.colorbar(CS, shrink=0.8, extend='both')
        plt.show()


def pdf(s1, s2, p, x, u1, y, u2):
    mul1_exp = -1 / (2 * (1 - p ** 2))
    mul2_exp = (x - u1) ** 2 / s1 ** 2 + (y - u2) ** 2 / s2 ** 2 - 2 * p * (x - u1) * (y - u2) / (s1 * s2)
    div = 2 * pi * s1 * s2 * sqrt(1 - p ** 2)
    return exp(mul1_exp * mul2_exp) / div
