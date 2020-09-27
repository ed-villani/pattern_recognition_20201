import warnings

import numpy as np

from commons.pdf import GaussianPDF, GaussianPDFTypes
from commons.solver import solver


def simple_classifier(points, classes):
    c = [
        np.argmax([
            GaussianPDF(GaussianPDFTypes.TWO_VAR)(
                x=p,
                d=d,
                p=0
            ) for d in classes]
        ) for p in points]
    return np.array(c)


def get_data_for_classification(classification, data):
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    return [
        data[np.where(classification == index)].T for index, c in enumerate(set(classification))
    ]


def data_frontier(data, grid, pdf, **kwargs):
    x = grid
    y = grid

    m = np.zeros((len(x), len(y)))
    solution = [solver(grid, pdf, d=d, **kwargs) for d in data]
    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            m[i][j] = np.argmax([s[i][j] for s in solution])
    return m
