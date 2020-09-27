import numpy as np
from numpy.random.mtrand import normal

from commons.classifiers import simple_classifier, get_data_for_classification, data_frontier
from commons.commons import join_data
from commons.pdf import PDF, PDFTypes
from commons.plotter import scatter_plot, surface_plot, frontier_plot
from commons.solver import solver


def main():
    mus = [
        [2, 2],
        [4, 4],
        [2, 4],
        [4, 2]
    ]
    sigmas = [0.6, 0.8, 0.2, 1]
    N = (100, 2)
    grid = np.arange(0.06, 6, 0.06)

    data = [normal(mu, sigma, size=N).T for mu, sigma in zip(mus, sigmas)]

    scatter_plot(data=data)

    classifiers = np.array([
        solver(
            grid,
            PDF(PDFTypes.TWO_VAR),
            d=d,
            p=0
        ) for index, d in enumerate(data)
    ])

    surface_plot(grid, grid, classifiers, z_limit=(0, 0.35))
    points = join_data(data)
    classes = simple_classifier(points, data)

    data_classified = get_data_for_classification(classes, points)
    scatter_plot(data=data_classified)
    frontier = data_frontier(data_classified, grid, PDF(PDFTypes.TWO_VAR), p=0)
    frontier_plot(data, grid, grid, frontier)


if __name__ == '__main__':
    main()
