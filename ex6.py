import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import normal

from fkm import FuzzyKMeans


def scatter_plot(data, colors, c=None, file_name=None, frontier = None):
    fig, ax = plt.subplots()
    plt.scatter(data[0], data[1], c=colors, alpha=0.5)
    if c is not None:
        plt.scatter(c[0], c[1], c='yellow')
    if frontier is not None:
        X, Y = np.meshgrid(frontier['grid_x'], frontier['grid_y'])
        ax.contour(Y, X, frontier['frontier'])
    if file_name is not None:
        fig.savefig(file_name)
    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    plt.show()



def main():
    sd = 0.7
    N_attr = 2
    total = 100
    mean = [2, 2]
    c_1 = np.array([normal(size=total) * sd + mean[i] for i in range(N_attr)]).T
    N_attr = 2
    total = 100
    mean = [4, 4]
    c_2 = np.array([normal(size=total) * sd + mean[i] for i in range(N_attr)]).T

    N_attr = 2
    total = 100
    mean = [2, 4]
    c_3 = np.array([normal(size=total) * sd + mean[i] for i in range(N_attr)]).T

    N_attr = 2
    total = 100
    mean = [4, 2]
    c_4 = np.array([normal(size=total) * sd + mean[i] for i in range(N_attr)]).T

    data_con = np.concatenate((c_1, c_2, c_3, c_4))
    np.random.shuffle(data_con)
    scatter_plot(data_con.T, 'black')
    ks = [2, 4, 8]
    for k in ks:

        fkm = FuzzyKMeans(data_con, k, 1e-19).fkm()
        # print(fkm)
        color = "bgrcmykw"
        colors = []
        for u in fkm[0]:
            max = np.argmax(u)
            c = color[max]
            if color[max] == 'w':
                c = 'purple'
            colors.append(c)
        scatter_plot(data_con.T, colors, fkm[1].T)
        print(fkm[1].T)


if __name__ == '__main__':
    main()
