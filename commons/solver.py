import numpy as np


def solver(grid, pdf, **kwargs):
    print(f'Calculating Solver')

    x = grid
    y = grid

    m = np.zeros((len(x), len(y)))

    for i, x_i in enumerate(x):
        for j, y_i in enumerate(y):
            m[i][j] = pdf(
                x=(x_i, y_i),
                **kwargs
            )
    return m
