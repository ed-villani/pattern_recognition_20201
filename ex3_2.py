import matplotlib.pyplot as plt
from numpy import ones, concatenate, meshgrid, cov, mean, array
from numpy.ma import arange
from numpy.random.mtrand import dirichlet

from ex3_1_new import data, pdf_2, data_frontier


def scatter_plot(random_data, label, frontier=False):
    fig, ax = plt.subplots()
    for d, l in zip(random_data, label):
        ax.scatter(d[0], d[1], edgecolors='none', label=l)
        ax.grid(True)

    if frontier:
        X, Y = meshgrid(frontier['grid_x'], frontier['grid_y'])
        ax.contour(Y, X, frontier['frontier'])

    ax.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


def main():
    training_percentage = 0.9
    x = dirichlet(ones(4), 1)[0] * training_percentage
    grid = arange(-2, 2, 0.04)
    N = 400
    sd = 0.3
    means = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

    c_1_training, c_1_random = data(2, N, sd, means[0], 0.9)
    c_2_training, c_2_random = data(2, N, sd, means[1], 0.9)
    c_3_training, c_3_random = data(2, N, sd, means[2], 0.9)
    c_4_training, c_4_random = data(2, N, sd, means[3], 0.9)

    c1_training = concatenate((c_1_training, c_4_training)).T
    c1_random = concatenate((c_1_random, c_4_random)).T

    c2_training = concatenate((c_2_training, c_3_training)).T
    c2_random = concatenate((c_2_random, c_3_random)).T


    data_con = concatenate((c1_random.T, c2_random.T))
    pdf_c1 = [
        pdf_2(2, cov(c1_training), d, mean(c1_training, axis=1)) for d in data_con
    ]
    pdf_c2 = [
        pdf_2(2, cov(c2_training), d, mean(c2_training, axis=1)) for d in data_con
    ]
    total = concatenate((c1_training, c2_training)).shape[0]
    n_c1 = c1_training.shape[0] / total
    n_c2 = 1 - n_c1

    final_c1 = []
    final_c2 = []
    for x, y, d in zip(pdf_c1, pdf_c2, data_con):
        k = (x * n_c1) / (y * n_c2)
        if k < 1:
            final_c1.append(d)
        else:
            final_c2.append(d)
    final_c1 = array(final_c1)
    final_c2 = array(final_c2)

    print(f"Erro: {((final_c1.shape[0] - c1_random.T.shape[0])/(c1_random.T.shape[0]*2))*100}%")

    scatter_plot([c1_training, c2_training]
                 , ['C1', 'C2'])
    scatter_plot([c1_training, c2_training, data_con.T]
                 , ['C1', 'C2', 'New Data'])
    scatter_plot([c1_random, c2_random]
                 , ['Should be C1', 'Should be C2'])
    scatter_plot([c1_training, c2_training, final_c1.T, final_c2.T]
                 , ['C1', 'C2', 'New C1', 'New C2'])

    c1_all = concatenate((c1_training.T, final_c1)).T
    c2_all = concatenate((c2_training.T, final_c2)).T

    frontier = {
        'grid_x': grid,
        'grid_y': grid,
        'frontier': data_frontier([c1_all, c2_all], grid)
    }

    scatter_plot([c1_all, c2_all]
                 , ['C1', 'C2'], frontier)

if __name__ == '__main__':
    main()
