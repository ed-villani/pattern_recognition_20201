import warnings

import numpy as np

from commons.pdf import PDF, PDFTypes
from commons.solver import solver


def bayesian_classifier(points, classes, pdf, type='simple', **kwargs):
    def prior_probability():
        n = None
        if type == 'simple':
            n = np.array([len(c.T) for c in classes])
        elif type == 'mix':
            n = np.array([sum([c.shape[1] for c in k]) for k in classes])
        total = sum(n)
        n = n / total
        return n

    n_per_classes = prior_probability()

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
            PDF(PDFTypes.TWO_VAR)(
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


def fuzzy_class_by_inner_class(fkm, data):
    from commons.fkm import FuzzyKMeans
    aux = [
        get_data_for_classification(FuzzyKMeans.normalize_membership_matrix(f[0]), c.T) for f, c in zip(fkm, data)
    ]
    for a in aux:
        for index, k in enumerate(a):
            if not k.shape[1]:
                a.pop(index)
    return aux


def data_frontier(data, grid, pdf, solution=None, **kwargs):
    x = grid
    y = grid

    m = np.zeros((len(x), len(y)))
    if solution is None:
        solution = [solver(grid, pdf, d=d, **kwargs) for d in data]
    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            m[i][j] = np.argmax([s[i][j] for s in solution])
    return m
