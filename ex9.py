import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import pinv, det
from sklearn.model_selection import KFold

from ex6 import scatter_plot
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


def pdf(n, K, x, m):
    from decimal import Decimal
    multiplier = 10 ** 16
    divider = multiplier ** K.shape[0]
    d = (1 / np.sqrt(Decimal(((2 * np.pi) ** n)) * Decimal(det(K * multiplier)) / Decimal(divider)))
    e = Decimal(np.exp(-(0.5 * ((x - m) @ pinv(K)) @ (x - m))))
    return float(d * e)


def solver(data, grid):
    x = grid
    y = grid

    m = np.zeros((len(x), len(y)))
    try:
        for i, x_i in enumerate(x):
            for j, y_i in enumerate(y):
                m[i][j] = pdf(
                    n=2,
                    K=np.cov(np.array(data).T),
                    x=(x_i, y_i),
                    m=np.mean(np.array(data).T, axis=1)
                )
    except Exception:
        i = 0
    return m


def main():
    np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    spiral = Spiral('ex7.txt')
    kf = KFold(n_splits=10)
    K = 30
    colors = random_colors(K + 1)
    colors_spiral = random_colors(2)
    accuracy = []
    i = 1
    better_accuracy = np.inf
    for train_index, test_index in kf.split(spiral.points):
        points_train = spiral.points[train_index]
        class_train = spiral.classification[train_index]

        points_train_1 = points_train[np.where(class_train == 1)]
        class_train_1 = class_train[np.where(class_train == 1)]

        points_train_2 = points_train[np.where(class_train == 2)]
        class_train_2 = class_train[np.where(class_train == 2)]

        points_test = spiral.points[test_index]
        class_test = spiral.classification[test_index]

        fkm_1 = FuzzyKMeans(points_train_1, int(K / 2), 1e-19).fkm()
        fkm_2 = FuzzyKMeans(points_train_2, int(K / 2), 1e-19).fkm()
        # fkm = k_means(K, points_train, 10000)
        # scatter_plot(
        #     points_train.T,
        #     [colors[int(np.argmax(u)) - 1] for u in fkm[0]],
        #     fkm[1].T,
        #     f'spiral_k_{K}.png'
        # )

        point_class_1, points_in_classes_1 = pointers_per_classes(int(K / 2), fkm_1, points_train_1)
        point_class_2, points_in_classes_2 = pointers_per_classes(int(K / 2), fkm_2, points_train_2)
        # c_1_clusters, c_2_clusters = get_clussters_per_class(K, point_class, class_train)

        p_x = len(points_train)
        p_1 = len(np.where(class_train == 1)[0]) / p_x
        p_2 = 1 - p_1
        final_result = []
        for p in points_test:
            pdf_1 = sum([pdf(len(p), np.cov(c.T), p, np.mean(c.T, axis=1)) if len(c) else 0 for c in
                         points_in_classes_1])
            pdf_2 = sum([pdf(len(p), np.cov(c.T), p, np.mean(c.T, axis=1)) if len(c) else 0 for c in
                         points_in_classes_2])

            if (p_1 * pdf_1) / (p_2 * pdf_2) >= 1:
                final_result.append(1)
            else:
                final_result.append(2)
        hit = 0
        for fr, r in zip(class_test, final_result):
            if r != fr:
                # print(f"Result: {fr}, Actual Class: {r}")
                hit = hit + 1
        # fold_accuracy = round((1 - abs(result * 2)) * 100, 2)
        print(f"{i} Hit: {hit}")

        if better_accuracy > hit:
            print(f"Gotta a smaller hit: {hit}")
            better_accuracy = hit
            # better_points = np.concatenate((spiral.points[train_index], spiral.points[test_index]))
            # better_classes = np.concatenate((spiral.classification[train_index], np.array(final_result)))
            better_points = spiral.points[test_index]
            better_classes = np.array(final_result)
            better_points_train = spiral.points[train_index]
            better_classes_train = spiral.classification[train_index]
            better_class_test = spiral.classification[test_index]
            better_points_test = spiral.points[test_index]
        accuracy.append(hit)
        i = i + 1
    print(f"Mean: {np.mean(accuracy)}")
    print(f"SD: {np.std(accuracy)}")

    # scatter_plot(
    #     better_points.T,
    #     [colors_spiral[int(u) - 1] for u in better_classes],
    #     file_name='spiral_original.png'
    # )

    scatter_plot(
        better_points_test.T,
        [colors_spiral[int(u) - 1] for u in better_class_test],
        file_name='spiral_original.png'
    )

    scatter_plot(
        better_points_train.T,
        [colors_spiral[int(u) - 1] for u in better_classes_train],
        file_name='spiral_original.png'
    )

    points = np.concatenate((better_points_train, better_points))
    classes = np.concatenate((better_classes_train, better_classes))
    scatter_plot(
        points.T,
        [colors_spiral[int(u) - 1] for u in classes],
        file_name='spiral_original.png'
    )
    points_in_classes = [[] for _ in range(2)]
    for index, point in enumerate(points):
        points_in_classes[int(classes[index]) - 1].append(
            point
        )

    frontier = {
        'grid_x': np.arange(-1, 1, 0.02),
        'grid_y': np.arange(-1, 1, 0.02),
        'frontier': data_frontier(points_in_classes, np.arange(-1, 1, 0.02))
    }

    scatter_plot_frontier(points_in_classes, ['C1', 'C2'], frontier=frontier)


def data_frontier(random_data, grid):
    x = grid
    y = grid

    m = np.zeros((len(x), len(y)))
    solver_k = np.array([solver(data, grid) for data in random_data])
    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            values_list = [s[i][j] for s in solver_k]
            m[i][j] = values_list.index(max(values_list))
    return m


def scatter_plot_frontier(random_data, label, frontier=False):
    fig, ax = plt.subplots()
    for d, l in zip(random_data, label):
        d = np.array(d).T
        ax.scatter(d[0], d[1], edgecolors='none', label=l)
        ax.grid(True)

    if frontier:
        X, Y = np.meshgrid(frontier['grid_x'], frontier['grid_y'])
        ax.contour(Y, X, frontier['frontier'])

    ax.legend()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


def pointers_per_classes(K, fkm, data):
    point_class = []
    points_in_classes = [[] for _ in range(K)]
    for index, point, in enumerate(data):
        points_in_classes[int(np.argmax(fkm[0][index])) - 1].append(
            point)
        point_class.append(int(np.argmax(fkm[0][index])))

    for index, c in enumerate(points_in_classes):
        points_in_classes[index] = np.array(c)
    return point_class, np.array(points_in_classes, dtype=object)


def get_clussters_per_class(K, point_class, data):
    class_1_k = []
    class_2_k = []
    df = pd.DataFrame(np.array([data, point_class]).T, columns=['class', 'cluster'])
    df = df.groupby(['class', 'cluster']).size().reset_index()
    for i in range(K):
        if df[(df['cluster'] == i) & (df['class'] == 1)].empty:
            class_2_k.append(i)
        elif df[(df['cluster'] == i) & (df['class'] == 2)].empty:
            class_1_k.append(i)
        else:
            class_1_val = df[(df['cluster'] == i) & (df['class'] == 1)].iloc[0][0]
            class_2_Val = df[(df['cluster'] == i) & (df['class'] == 2)].iloc[0][0]

            if class_1_val > class_2_Val:
                class_1_k.append(i)
            else:
                class_2_k.append(i)
    return class_1_k, class_2_k


if __name__ == '__main__':
    main()
