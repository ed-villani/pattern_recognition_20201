import warnings

from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from ex9 import Spiral, random_colors
from commons.pdf import pdf


def solver(grid, data):
    x = grid
    y = grid

    m = np.zeros((len(x), len(y)))

    K = np.cov(np.array(data).T)
    mean = np.mean(np.array(data).T, axis=1)

    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            m[i][j] = pdf(
                n=data.shape[1],
                K=K,
                x=(x_i, y_i),
                m=mean
            )
    return m


# def data_frontier(data, grid):
#     x = grid
#     y = grid
#
#     m = np.zeros((len(x), len(y)))
#     solution = [solver(grid, datum) for datum in data]
#     for i, x_i in enumerate(x):
#         for j, y_i in enumerate(y):
#             values_list = [s[i][j] for s in solution]
#             m[i][j] = values_list.index(max(values_list))
#     return m


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    spiral = Spiral('data/spiral.txt')

    X_train, X_test, y_train, y_test = train_test_split(spiral.data, spiral.classification, test_size=0.1,
                                                        random_state=1)
    colors = random_colors(3)

    fig, ax = plt.subplots()
    spiral_1 = np.where(X_train.T[-1] == 1)
    spiral_2 = np.where(X_train.T[-1] == 2)
    ax.scatter(X_train[spiral_1].T[1], X_train[spiral_1].T[2], alpha=0.3, edgecolors='none', c=colors[0])
    ax.scatter(X_train[spiral_2].T[1], X_train[spiral_2].T[2], alpha=0.3, edgecolors='none', c=colors[1])
    plt.show()

    C = 0.75
    svm_linear = svm.SVC(kernel='rbf', C=C * 10 ** 3)
    svm_linear.fit(X_train, y_train)

    fig, ax = plt.subplots()
    ax.scatter(X_train[spiral_1].T[1], X_train[spiral_1].T[2], alpha=0.3, edgecolors='none', c=colors[0])
    ax.scatter(X_train[spiral_2].T[1], X_train[spiral_2].T[2], alpha=0.3, edgecolors='none', c=colors[1])
    ax.scatter(X_train[svm_linear.support_].T[1], X_train[svm_linear.support_].T[2], alpha=0.5, edgecolors='none',
               c='black')
    plt.show()
    fig, ax = plt.subplots()
    predicted = svm_linear.predict(X_test)
    hit = 0
    for x, y in zip(predicted, y_test):
        if x != y:
            hit += 1
    print(f"C: {C * 10 ** 3}\nFails: {hit}")

    grid = np.arange(-7, 7, 14/100)
    # frontier = data_frontier([X_train[spiral_1][:, 1:3].T, X_train[spiral_2][:, 1:3].T], grid)
    random_data = [X_train[spiral_1][:, 1:3].T, X_train[spiral_2][:, 1:3].T]
    fig, ax = plt.subplots()
    for data in random_data:
        X, Y = np.meshgrid(grid, grid)
        CS = ax.contour(Y, X, solver(grid, data.T))
    fig.colorbar(CS, shrink=0.8, extend='both')
    plt.show()


if __name__ == '__main__':
    main()
