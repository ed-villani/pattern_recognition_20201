from random import shuffle

import numpy as np
from sklearn.model_selection import KFold

from ex9 import Spiral


def spread(data):
    return max(1.06 * np.std(data, axis=0) * data.shape[0] ** (-1 / 5))


def kde(data, h):
    def constant(data, h):
        N = data.shape[0]
        n = data.shape[1]
        return 1 / (N * ((np.sqrt(2 * np.pi) * h) ** n))

    def kde_exp(x, xi):
        upper = (x - xi) ** 2
        lower = 2 * h ** 2
        return np.prod(np.exp(-(upper / lower)))

    def summation_kde(x):
        return constant(data, h) * sum([kde_exp(x, xi) for xi in data])

    return summation_kde


def main():
    spiral = Spiral('data/spiral.txt')
    data = np.insert(spiral.points, 0, spiral.classification, axis=1)
    shuffle(data)
    data = data[:, 1:]
    classes = data[:, 0]

    kf = KFold(n_splits=10)
    n_tests = 9
    h_base = spread(spiral.points)
    step = h_base / (n_tests - 1)
    all_h = [(h_base * 1.5) - (step * i) for i in range(n_tests)]
    better_h = 0
    for h in all_h:
        hits = []
        better_hit = np.inf
        for train_index, test_index in kf.split(data):
            points_train = spiral.points[train_index]
            class_train = spiral.classification[train_index]

            points_train_1 = points_train[np.where(class_train == 1)]
            class_train_1 = class_train[np.where(class_train == 1)]

            points_train_2 = points_train[np.where(class_train == 2)]
            class_train_2 = class_train[np.where(class_train == 2)]

            points_test = spiral.points[test_index]
            class_test = spiral.classification[test_index]
            hit_bayes = 0
            total = points_train.shape[0]
            p1 = points_train_1.shape[0] / total
            p2 = 1 - p1
            result = []
            for index, p in enumerate(points_test):
                k1 = kde(points_train_1, h)(p)
                k2 = kde(points_train_2, h)(p)
                # print(k1, k2, 1 if k1 > k2 else 2, class_test[index])
                classe_baye = 1 if (k1 * p1) / (k2 * p2) >= 1 else 2
                result.append(classe_baye)
                if classe_baye != class_test[index]:
                    hit_bayes = hit_bayes + 1

            if hit_bayes < better_hit:
                print(f"Better Hit: {hit_bayes}")
                better = result
                better_hit = hit_bayes

            hits.append(hit_bayes)
            accuracy = np.mean(hits)
            if accuracy > better_h:
                best_result = better
                mejor_h = h
                print(mejor_h)
                better_h = accuracy


if __name__ == '__main__':
    main()
