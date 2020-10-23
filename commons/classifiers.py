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
            try:
                pdfs_values[index].append(
                    pdf(
                        x=p,
                        d=c,
                        **kwargs
                    )
            )
            except Exception:
                i=0
    pdfs_values = np.array(pdfs_values).reshape(len(pdfs_values), len(points)).T

    classification = [np.argmax(values * n_per_classes) for values in pdfs_values]
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


def get_data_for_classification(classification, data, n_classes=None):
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    if n_classes is not None:
        values = enumerate(list(range(n_classes)))
    else:
        values = enumerate(set(classification))

    return [
            data[np.where(classification == index)].T for index, c in enumerate(list(values))
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
