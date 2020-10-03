from datetime import datetime

import numpy as np
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from commons.commons import calculate_accuracy_percentage, save_to_csv
from commons.plotter import line_plot


def main():
    n_components = 7
    n_splits = 10

    cancer = load_breast_cancer()
    points, classification = cancer.data, cancer.target

    # for n in range(n_components):
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(points)

    transformed_points = pca.fit_transform(points)
    accuracies_list = []
    i = 0
    for train_index, test_index in KFold(n_splits=n_splits, shuffle=True).split(transformed_points):
        X_train = points[train_index]
        y_train = classification[train_index]

        X_test = points[test_index]
        y_test = classification[test_index]

        C = 15
        svm_linear = svm.SVC(kernel='rbf', C=C, gamma=1 / (np.std(X_train)))
        svm_linear.fit(X_train, y_train)

        svm_classification = svm_linear.predict(X_test)
        accuracy = calculate_accuracy_percentage(y_test, svm_classification)
        i = i + 1
        accuracies_list.append([i, round(accuracy, 4), 0.9, C, n_components, datetime.now().strftime("%Y/%m/%dT%H:%M:%S")])
    save_to_csv('/Users/eduardovillani/git/pattern_recognition_20201/data/ex_14.csv', accuracies_list)
    line_plot(list(range(len(pca.explained_variance_))), pca.explained_variance_)


if __name__ == '__main__':
    main()
