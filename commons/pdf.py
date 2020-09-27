from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.linalg import pinv, det


@dataclass
class GaussianPDFTypes:
    MULTI_VAR = 'multi-var'
    TWO_VAR = 'two-var'


class GaussianPDF:
    def __new__(cls, type='multi-var'):
        if type == GaussianPDFTypes.MULTI_VAR:
            return GaussianPDF.pdf
        elif type == GaussianPDFTypes.TWO_VAR:
            return GaussianPDF.pdf_2_var
        else:
            raise Exception

    @staticmethod
    def pdf_2_var(s1, s2, p, x, u1, u2):
        mul1_exp = -1 / (2 * (1 - p ** 2))
        mul2_exp = (x[0] - u1) ** 2 / s1 ** 2 + (x[1] - u2) ** 2 / s2 ** 2 - 2 * p * (x[0] - u1) * (x[1] - u2) / (s1 * s2)
        div = 2 * np.pi * s1 * s2 * np.sqrt(1 - p ** 2)
        return np.exp(mul1_exp * mul2_exp) / div

    @staticmethod
    def pdf(n, K, x, m):
        multiplier = 10 ** 16
        divider = multiplier ** K.shape[0]
        d = (1 / np.sqrt(Decimal(((2 * np.pi) ** n)) * Decimal(det(K * multiplier)) / Decimal(divider)))
        e = Decimal(np.exp(-(0.5 * ((x - m) @ pinv(K)) @ (x - m))))
        return float(d * e)
