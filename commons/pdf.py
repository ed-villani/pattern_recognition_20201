import warnings
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.linalg import pinv, det


@dataclass
class PDFTypes:
    MULTI_VAR = 'multi-var'
    TWO_VAR = 'two-var'
    MIXTURE = 'mixture'
    KDE = 'kde'


class PDF:
    def __new__(cls, type='multi-var'):
        if type == PDFTypes.MULTI_VAR:
            return PDF.pdf
        elif type == PDFTypes.TWO_VAR:
            return PDF.pdf_2_var
        elif type == PDFTypes.MIXTURE:
            return PDF.gaussian_mixture
        elif type == 'kde':
            return PDF.kde
        else:
            raise Exception

    @staticmethod
    def pdf_2_var(d, p, x):
        s = np.std(d, axis=1)
        u = np.mean(d, axis=1)
        mul1_exp = -1 / (2 * (1 - p ** 2))
        mul2_exp = (x[0] - u[0]) ** 2 / s[0] ** 2 + (x[1] - u[1]) ** 2 / s[1] ** 2 - 2 * p * (x[0] - u[0]) * (
                x[1] - u[1]) / (s[0] * s[1])
        div = 2 * np.pi * s[0] * s[1] * np.sqrt(1 - p ** 2)
        return np.exp(mul1_exp * mul2_exp) / div

    @staticmethod
    def pdf(x, d):
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        def calc_det():
            multiplier = 10 ** 19
            divider = multiplier ** K.shape[0]
            return Decimal(det(K * multiplier)) / Decimal(divider)

        K = np.cov(d)
        m = np.mean(d, axis=1)
        n = K.shape[0]
        d = (1 / np.sqrt(Decimal(((2 * np.pi) ** n)) * calc_det()))
        e = Decimal(np.exp(-(0.5 * ((x - m) @ pinv(K)) @ (x - m))))
        return float(d * e)

    @staticmethod
    def gaussian_mixture(x, d):
        return sum([PDF.pdf(x, data) for data in d])

    @staticmethod
    def kde_spread(data):
        return max(1.06 * np.std(data, axis=0) * data.shape[0] ** (-1 / 5))

    @staticmethod
    def kde(x, d, h=None):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if h is None:
            h = PDF.kde_spread(d)

        def constant(data, h):
            N = data.shape[0]
            n = data.shape[1]
            return 1 / (n * ((np.sqrt(2 * np.pi) * h) ** N))

        def kde_exp(p, xi):
            upper = (p - xi) ** 2
            lower = 2 * h ** 2
            return np.prod(np.exp(-(upper / lower)))

        def summation_kde(p):
            return constant(d, h) * sum([kde_exp(p, xi) for xi in d.T])

        return summation_kde(x)

    # @staticmethod
    # def generate_kde_pdf(x, d, h=None):
    #     total_samples = d.shape[0]
    #     dimensions = d.shape[1]
    #     if h is None:
    #         h = max(1.06 * d.std() * total_samples ** (-1 / 5), 0.00001)
    #     multiplier = 1 / (total_samples * (((2 * np.pi) * (1 / 2) * h) * dimensions))
    #
    #     def pdf(*x):
    #         x_array = np.array(x)
    #         all_terms = np.e * -(((x_array - d.T) * 2) / (2 * h ** 2))
    #         return multiplier * all_terms.prod(axis=1).sum()
    #
    #     return pdf(x)
