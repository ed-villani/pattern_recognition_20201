import warnings

import numpy as np

from commons.pdf import GaussianPDF, GaussianPDFTypes
from commons.solver import solver


def bayesian_classifier(points, classes, pdf, **kwargs):
    n_per_classes = np.array([len(c.T) for c in classes])
    total = sum(n_per_classes)
    n_per_classes = n_per_classes / total
    pdfs_values = []
    for index, c in enumerate(classes):
        pdfs_values.append([])
        for p in points:
            pdfs_values[index].append(
                pdf(
                    x=p,
                    d=c,
                    **kwargs
                )
            )

    classification = []
    for index in range(len(points)):
        k = (pdfs_values[0][index] * n_per_classes[0]) / (pdfs_values[1][index] * n_per_classes[1])
        if k > 1:
            classification.append(0)
        else:
            classification.append(1)
    return np.array(classification)


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
