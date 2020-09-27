import matplotlib.pyplot as plt
import numpy as np

import logging

logging.getLogger().setLevel(logging.CRITICAL)


def frontier_plot(data, x, y, frontier):
    fig, ax = plt.subplots()
    for d in data:
        ax.scatter(d[0], d[1], alpha=0.3, edgecolors='none')
        ax.legend()
        ax.grid(True)
    X, Y = np.meshgrid(x, y)
    ax.contour(Y, X, frontier)
    plt.show()


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
