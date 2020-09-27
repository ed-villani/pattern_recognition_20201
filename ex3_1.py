import numpy as np
from sklearn.model_selection import train_test_split

from commons.classifiers import bayesian_classifier, get_data_for_classification, data_frontier
from commons.commons import join_data, gen_data, calculate_accuracy_percentage, join_classification
from commons.pdf import GaussianPDF, GaussianPDFTypes
from commons.plotter import scatter_plot, frontier_plot


def main():
    mus = [
        [2, 2],
        [4, 4]
    ]
    sigmas = [0.8, 0.4]
    N = (200, 2)
    grid = np.arange(0.06, 6, 0.06)
    classification = [0, 1]
    ylim = (-1, 6)
    xlim = (-1, 6)
    data = [gen_data(mu, sigma, N, c) for mu, sigma, c in zip(mus, sigmas, classification)]
    scatter_plot(data=data, ylim=ylim, xlim=xlim)

    points = join_data(data)
    X_train, X_test, y_train, y_test = train_test_split(points[:, :-1], points[:, -1], train_size=0.9)

    X_train_per_class = get_data_for_classification(y_train, X_train)
    scatter_plot(data=X_train_per_class, ylim=ylim, xlim=xlim)
    scatter_plot(data=X_test, ylim=ylim, xlim=xlim)

    classification = bayesian_classifier(
        X_test, X_train_per_class, GaussianPDF(GaussianPDFTypes.TWO_VAR), p=0
    )
    calculate_accuracy_percentage(y_test, classification)

    data_classified = get_data_for_classification(*join_classification(X_train, y_train, X_test, classification))
    scatter_plot(data=data_classified, ylim=ylim, xlim=xlim)

    frontier = data_frontier(data_classified, grid, GaussianPDF(GaussianPDFTypes.TWO_VAR), p=0)
    frontier_plot(data, grid, grid, frontier, ylim=ylim, xlim=xlim)


if __name__ == '__main__':
    main()
