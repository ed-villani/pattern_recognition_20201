import numpy as np

from commons.classifiers import get_data_for_classification
from commons.fkm import FuzzyKMeans
from commons.pdf import PDF, PDFTypes
from commons.plotter import scatter_plot, surface_plot
from commons.solver import solver
from commons.spiral import Spiral


def main():
    K = 30
    ylim = (-1.1, 1.1)
    xlim = (-1.1, 1.1)
    grid = np.arange(-1, 1, 0.02)
    points, classification = Spiral().points, Spiral().classification
    scatter_plot(
        get_data_for_classification(classification, points),
        ylim=ylim,
        xlim=xlim
    )
    fkm = FuzzyKMeans(points, K, 1e-19).fkm()
    classification = FuzzyKMeans.normalize_membership_matrix(fkm[0])
    data_classified = get_data_for_classification(classification, points)
    scatter_plot(
        data_classified,
        centers=fkm[1],
        title=f'Clusters for K = {K}',
        ylim=ylim,
        xlim=xlim
    )
    classifiers = np.array([
        solver(
            grid,
            GaussianPDF(PDFTypes.MULTI_VAR),
            d=d
        ) for index, d in enumerate(data_classified)
    ])

    surface_plot(grid, grid, classifiers)


if __name__ == '__main__':
    main()
