from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold

from commons.classifiers import get_data_for_classification, bayesian_classifier
from commons.commons import calculate_accuracy_percentage, save_to_csv
from commons.pdf import GaussianPDF, GaussianPDFTypes
from ex9_1 import Spiral


def main():
    n_splits = 10
    n_tests = 9
    spiral = Spiral()
    points, classification = spiral.points, spiral.classification

    h_base = GaussianPDF.kde_spread(points)
    step = h_base / (n_tests - 1)
    all_h = [(h_base * 1.5) - (step * i) for i in range(n_tests)]
    better_h = None
    better_accuracy = 0
    accuracies_list = []
    i = 0
    for h in all_h:
        for train_index, test_index in KFold(n_splits=n_splits).split(points):
            points_train = points[train_index]
            class_train = classification[train_index]

            points_test = points[test_index]
            class_test = classification[test_index]

            data_classified = get_data_for_classification(class_train, points_train)

            bayesian_classification = bayesian_classifier(
                points_test,
                data_classified,
                GaussianPDF(GaussianPDFTypes.KDE),
                h=h
            )

            accuracy = calculate_accuracy_percentage(class_test, bayesian_classification)
            accuracies_list.append([i, round(accuracy, 4), 0.9, h, datetime.now().strftime("%Y/%m/%dT%H:%M:%S")])

            i += 1

            if accuracy > better_accuracy:
                better_accuracy = accuracy
                better_h = h
                print(f'Got a better spread: {better_h}')

                better_points_train = points_train
                better_class_train = class_train
                better_points_test = points_test
                better_class_test = bayesian_classification

                better_points, better_classification = np.concatenate(
                    (better_points_train, better_points_test)), np.concatenate(
                    (better_class_train, better_class_test))

    save_to_csv('/Users/eduardovillani/git/pattern_recognition_20201/data/ex_11.csv', accuracies_list)


if __name__ == '__main__':
    main()
