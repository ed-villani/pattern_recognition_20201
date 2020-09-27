import numpy as np
from numpy.random.mtrand import normal

from commons.pdf import PDFTypes, PDF
from commons.plotter import surface_plot, contour_plot, scatter_plot
from commons.solver import solver


def main():
    mus = [
        [2, 2],
        [4, 4]
    ]
    sigmas = [0.6, 0.6]
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

    surface_plot(grid, grid, classifiers)
    contour_plot(grid, grid, classifiers)


if __name__ == '__main__':
    main()
