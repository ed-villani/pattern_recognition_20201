from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold

from commons.classifiers import get_data_for_classification, fuzzy_class_by_inner_class, bayesian_classifier, \
    data_frontier
from commons.commons import calculate_accuracy_percentage, save_to_csv
from commons.fkm import FuzzyKMeans
from commons.pdf import PDF, PDFTypes
from commons.plotter import surface_plot, contour_plot, frontier_plot
from commons.solver import solver
from commons.spiral import Spiral


def main():
    better_fuzzy_data, better_classification, better_points = None, None, None
    K = 30
    n_splits = 10
    ylim = (-1.1, 1.1)
    xlim = (-1.1, 1.1)
    grid = np.arange(-1, 1, 0.02)
    spiral = Spiral()
    points, classification = spiral.points, spiral.classification
    better_accuracy = 0
    accuracies_list = []
    i = 0
    for train_index, test_index in KFold(n_splits=n_splits).split(points):
        points_train = points[train_index]
        class_train = classification[train_index]

        points_test = points[test_index]
        class_test = classification[test_index]

        data_classified = get_data_for_classification(class_train, points_train)
        fkm = [FuzzyKMeans(c.T, int(K / 2), 1e-19).fkm() for c in data_classified]

        fuzzy_data_classified = fuzzy_class_by_inner_class(fkm, data_classified)
        bayesian_classification = bayesian_classifier(
            points_test,
            fuzzy_data_classified,
            PDF(PDFTypes.MIXTURE),
            'mix'
        )
        accuracy = calculate_accuracy_percentage(class_test, bayesian_classification)
        accuracies_list.append([i, round(accuracy, 4), 0.9, datetime.now().strftime("%Y/%m/%dT%H:%M:%S")])

        i += 1

        if accuracy > better_accuracy:
            better_points_train = points_train
            better_class_train = class_train
            better_points_test = points_test
            better_class_test = bayesian_classification

            better_points, better_classification = np.concatenate(
                (better_points_train, better_points_test)), np.concatenate(
                (better_class_train, better_class_test))
            better_fuzzy_data = fuzzy_data_classified

    save_to_csv('/Users/eduardovillani/git/pattern_recognition_20201/data/ex_9.csv', accuracies_list)

    # data_classified = get_data_for_classification(better_classification, better_points)

    classifiers = np.array([
        solver(
            grid,
            PDF(PDFTypes.MIXTURE),
            d=d
        ) for index, d in enumerate(better_fuzzy_data)
    ])

    surface_plot(grid, grid, classifiers)
    contour_plot(grid, grid, classifiers)
    frontier = data_frontier(better_fuzzy_data, grid, PDF(PDFTypes.MIXTURE), solution=classifiers)
    frontier_plot(better_points, grid, grid, frontier)


if __name__ == '__main__':
    main()
