import pandas as pd
import numpy as np


class XAndC:
    def __new__(cls):
        df = pd.read_excel('/Users/eduardovillani/git/pattern_recognition_20201/data/x_and_c.xlsx', header=None)
        df = np.array(df)

        return df, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]

    @staticmethod
    def filters():
        f = np.reshape(np.array([1, -1, -1, -1, 1, -1, -1, -1, 1]), (3, 3))
        f2 = np.reshape(np.array([1, -1, 1, -1, 1, -1, 1, -1, 1]), (3, 3))
        f3 = np.reshape(np.array([1, 1, 1, 1, -1, -1, 1, -1, -1]), (3, 3))
        # f4 = np.reshape(np.array([1, -1, -1, 1, -1, -1, 1, 1, 1]), (3, 3))
        f4 = np.reshape(np.array([1, 1, 1, -1, -1, 1, -1, -1, 1]), (3, 3))
        return np.array([f, f2, f3, f4])
