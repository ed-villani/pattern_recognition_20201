import random
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from ex3_1 import pdf_2
from commons.fkm import FuzzyKMeans


class Spiral:
    def __init__(self, path):
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
        return deepcopy(self._spiral[:, 1:3])

    @property
    def classification(self):
        return deepcopy(self._spiral[:, 3])


def random_colors(number_of_colors):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            for _ in range(number_of_colors)]


def surface_plot(grid, random_data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for data in random_data:
        X, Y = np.meshgrid(grid, grid)
        CS = ax.plot_surface(
            X,
            Y,
            solver(data, grid),
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


def solver(data, grid):
    x = grid
    y = grid

    m = np.zeros((len(x), len(y)))

    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            m[i][j] = pdf_2(
                n=2,
                K=np.cov(np.array(data).T),
                x=(x_i, y_i),
                m=np.mean(np.array(data).T, axis=1)
            )
    return m


def main():
    K = 20
    colors = random_colors(K + 1)
    colors_spiral = random_colors(2)
    # scatter_plot(
    #     Spiral('ex7.txt').points.T,
    #     [colors_spiral[int(u) - 1] for u in Spiral('ex7.txt').classification],
    #     file_name='spiral_original.png'
    # )

    fkm = FuzzyKMeans(Spiral('ex7.txt').points, K, 1e-19).fkm()
    # scatter_plot(
    #     Spiral('ex7.txt').points.T,
    #     [colors[int(np.argmax(u)) - 1] for u in fkm[0]],
    #     fkm[1].T,
    #     f'spiral_k_{20}.png'
    # )

    points_in_classes = [[] for _ in range(K)]
    for index, point in enumerate(Spiral('ex7.txt').points):
        points_in_classes[int(np.argmax(fkm[0][index])) - 1].append(
            point
        )

    # pdf_per_classes = [[] for _ in range(K)]
    # for index, c in enumerate(points_in_classes):
    #     for p in Spiral('ex7.txt').points:
    #         pdf_per_classes[index].append(
    #             pdf_2(
    #                 n=2,
    #                 K=np.cov(np.array(c).T),
    #                 x=p,
    #                 m=np.mean(np.array(c).T, axis=1)
    #             )
    #         )
    surface_plot(np.arange(-1, 1, 0.02), points_in_classes)


if __name__ == '__main__':
    main()
