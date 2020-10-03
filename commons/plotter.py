import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import logging

logging.getLogger().setLevel(logging.CRITICAL)


def confusion_matrix(confusion_matrix, font_scale=0.5, inner_font_size=8):
    dpi = 200
    plt.figure(figsize=(800 / dpi, 800 / dpi), dpi=dpi)
    df_cm = pd.DataFrame(confusion_matrix)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=font_scale)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": inner_font_size})  # font size
    plt.show()


# def print_figs(data):
#     x = 1
#     y = 10
#     fig = plt.figure(figsize=(x, y))
#     for i in range(y*x):
#         ax = fig.add_subplot(x, y, i + 1)
#         ax.imshow(data.images[i], cmap=plt.cm.bone)
#     plt.show()


def frontier_plot(data, x, y, frontier, **kwargs):
    fig, ax = plt.subplots()
    for d in data:
        ax.scatter(d[0], d[1], alpha=0.3, edgecolors='none')
        ax.legend()
        ax.grid(True)
    X, Y = np.meshgrid(x, y)
    ax.contour(Y, X, frontier)

    if kwargs.get('xlim') is not None:
        plt.xlim(*kwargs['xlim'])
    if kwargs.get('ylim') is not None:
        plt.ylim(*kwargs['ylim'])

    plt.show()


def contour_plot(x, y, z, color_bar=False, **kwargs):
    fig, ax = plt.subplots()
    CS = None
    for d in z:
        X, Y = np.meshgrid(x, y)
        CS = ax.contour(Y, X, d)
        ax.clabel(CS, inline=1, fontsize=10)
    if color_bar:
        fig.colorbar(CS, shrink=0.8, extend='both')
    if kwargs.get('xlim') is not None:
        plt.xlim(*kwargs['xlim'])
    if kwargs.get('ylim') is not None:
        plt.ylim(*kwargs['ylim'])
    plt.show()


def line_plot(x, y):
    def set_labels():
        for x_i, y_i in zip(x, y):
            label = "{:.2f}".format(y_i)

            plt.annotate(label,  # this is the text
                         (x_i, y_i),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='left')  # horizontal alignment can be left, right or center
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, y, marker='o')
    set_labels()
    plt.yscale(value="log")
    plt.show()

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, y, marker='o')
    set_labels()
    plt.show()


def scatter_plot(data, **kwargs):
    fig, ax = plt.subplots()
    for index, d in enumerate(data):
        if kwargs.get('colors') is None:
            ax.scatter(d[0], d[1], alpha=0.3, edgecolors='none')
        else:
            ax.scatter(d[0], d[1], alpha=0.3, edgecolors='none', c=kwargs['colors'][index])
        ax.legend()
        ax.grid(True)

    if kwargs.get('support_vectors') is not None:
        for c in kwargs['support_vectors']:
            ax.scatter(c[0], c[1], edgecolors='none', c='green')
    if kwargs.get('centers') is not None:
        for c in kwargs['centers']:
            ax.scatter(c[0], c[1], edgecolors='none', c='black')
    if kwargs.get('xlim') is not None:
        plt.xlim(*kwargs['xlim'])
    if kwargs.get('ylim') is not None:
        plt.ylim(*kwargs['ylim'])
    if kwargs.get('title') is not None:
        plt.title(kwargs['title'])

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
