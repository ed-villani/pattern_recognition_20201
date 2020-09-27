import warnings
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.linalg import pinv, det


@dataclass
class GaussianPDFTypes:
    MULTI_VAR = 'multi-var'
    TWO_VAR = 'two-var'
    MIXTURE = 'MIXTURE'


class GaussianPDF:
    def __new__(cls, type='multi-var'):
        if type == GaussianPDFTypes.MULTI_VAR:
            return GaussianPDF.pdf
        elif type == GaussianPDFTypes.TWO_VAR:
            return GaussianPDF.pdf_2_var
        elif type == GaussianPDFTypes.MIXTURE:
            return GaussianPDF.gaussian_mixture
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
        return sum([GaussianPDF.pdf(x, data) for data in d])
