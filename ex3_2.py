import numpy as np
from sklearn.model_selection import train_test_split

from commons.classifiers import get_data_for_classification, bayesian_classifier, data_frontier
from commons.commons import gen_data, join_data, calculate_accuracy_percentage, join_classification
from commons.pdf import PDF, PDFTypes
from commons.plotter import scatter_plot, frontier_plot


def main():
    mus = [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ]
    sigmas = [0.3, 0.3, 0.3, 0.3]
    N = (200, 2)
    grid = np.arange(-2, 2, 0.04)
    classification = [0, 1, 1, 0]
    ylim = (-2, 2)
    xlim = (-2, 2)
    data = [gen_data(mu, sigma, N, c) for mu, sigma, c in zip(mus, sigmas, classification)]
    points = join_data(data, True)

    scatter_plot(
        data=get_data_for_classification(points[:, -1], points[:, :-1]),
        ylim=ylim,
        xlim=xlim
    )
    X_train, X_test, y_train, y_test = train_test_split(points[:, :-1], points[:, -1], train_size=0.9)
    X_train_per_class = get_data_for_classification(y_train, X_train)
    scatter_plot(data=X_train_per_class, ylim=ylim, xlim=xlim)
    scatter_plot(data=get_data_for_classification(y_test, X_test), ylim=ylim, xlim=xlim)
    classification = bayesian_classifier(
        X_test, X_train_per_class, PDF(PDFTypes.MULTI_VAR)
    )
    calculate_accuracy_percentage(y_test, classification)
    data_classified = get_data_for_classification(*join_classification(X_train, y_train, X_test, classification))
    scatter_plot(data=data_classified, ylim=ylim, xlim=xlim)

    frontier = data_frontier(data_classified, grid, PDF(PDFTypes.MULTI_VAR))
    frontier_plot(data_classified, grid, grid, frontier, ylim=ylim, xlim=xlim)


if __name__ == '__main__':
    main()
