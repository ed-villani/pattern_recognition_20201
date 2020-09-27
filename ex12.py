import warnings

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from commons.classifiers import get_data_for_classification, data_frontier, fuzzy_class_by_inner_class
from commons.commons import calculate_accuracy_percentage
from commons.fkm import FuzzyKMeans
from commons.pdf import PDF, PDFTypes
from commons.plotter import scatter_plot, surface_plot, frontier_plot, contour_plot
from commons.solver import solver
from commons.spiral import Spiral


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    K = 30
    train_size = 0.9
    spiral = Spiral()
    grid = np.arange(-2, 2, 0.02)
    ylim = (-1.1, 1.1)
    xlim = (-1.1, 1.1)
    points, classification = spiral.points, spiral.classification

    X_train, X_test, y_train, y_test = train_test_split(points, classification, train_size=train_size)

    data_classified = get_data_for_classification(y_train, X_train)
    scatter_plot(data=data_classified)

    C = 15
    svm_linear = svm.SVC(kernel='rbf', C=C, gamma=1 / (np.std(X_train)))
    svm_linear.fit(X_train, y_train)
    scatter_plot(data=data_classified, support_vectors=svm_linear.support_vectors_)

    classification = svm_linear.predict(X_test)
    calculate_accuracy_percentage(y_test, classification)

    data_classified = get_data_for_classification(np.concatenate((y_train, classification)),
                                                  np.concatenate((X_train, X_test)))
    scatter_plot(data=data_classified)

    fkm = [FuzzyKMeans(c.T, int(K / 2), 1e-19).fkm() for c in data_classified]

    fuzzy_data_classified = fuzzy_class_by_inner_class(fkm, data_classified)

    classifiers = np.array([
        solver(
            grid,
            PDF(PDFTypes.MIXTURE),
            d=d
        ) for index, d in enumerate(fuzzy_data_classified)
    ])

    surface_plot(grid, grid, classifiers)
    contour_plot(grid, grid, classifiers, xlim=xlim, ylim=ylim)
    frontier = data_frontier(fuzzy_data_classified, grid, PDF(PDFTypes.MIXTURE), solution=classifiers)
    frontier_plot(data_classified, grid, grid, frontier)


if __name__ == '__main__':
    main()
