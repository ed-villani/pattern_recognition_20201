import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import normal

from commons.pdf import GaussianPDFTypes, GaussianPDF
from commons.solver import solver


def contour_plot(x, y, z, color_bar=False):
    fig, ax = plt.subplots()
    CS = None
    for d in z:
        X, Y = np.meshgrid(x, y)
        CS = ax.contour(Y, X, d)
        ax.clabel(CS, inline=1, fontsize=10)
    if color_bar:
        fig.colorbar(CS, shrink=0.8, extend='both')
    plt.show()


def scatter_plot(data):
    fig, ax = plt.subplots()
    for d in data:
        ax.scatter(d[0], d[1], alpha=0.3, edgecolors='none')
        ax.legend()
        ax.grid(True)
    plt.show()


def surface_plot(x, y, z, **kwargs):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    CS = None
    for d in z:
        X, Y = np.meshgrid(x, y)
        CS = ax.plot_surface(
            X,
            Y,
            d,
            rstride=1,
            cstride=1,
            cmap='viridis',
            edgecolor='none'
        )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if kwargs.get('z_limit') is not None:
        ax.set_zlim(*kwargs['z_limit'])
    fig.colorbar(CS, shrink=0.8, extend='both')
    plt.show()


def main():
    mus = [
        [2, 2],
        [4, 4]
    ]
    sigmas = [0.6, 0.6]
    N = (100, 2)
    grid = np.arange(0.06, 6, 0.06)

    data = [normal(mu, sigma, size=N).T for mu, sigma in zip(mus, sigmas)]
    scatter_plot(data=data)

    classifiers = np.array([
        solver(
            grid,
            GaussianPDF(GaussianPDFTypes.TWO_VAR),
            s1=np.std(d, axis=1)[0],
            s2=np.std(d, axis=1)[1],
            p=0,
            u1=np.mean(d, axis=1)[0],
            u2=np.mean(d, axis=1)[1]
        ) for index, d in enumerate(data)
    ])

    surface_plot(grid, grid, classifiers)
    contour_plot(grid, grid, classifiers)


if __name__ == '__main__':
    main()
