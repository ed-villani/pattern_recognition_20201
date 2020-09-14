import numpy as np
from numpy.random.mtrand import normal
from scipy.spatial.distance import cdist

from ex6 import scatter_plot
from ex7 import random_colors


def groups(data, k):
    clusters = np.arange(data.shape[0])
    while np.unique(clusters).shape[0] > k:
        distance = cdist(data, data)
        distance[distance == 0] = np.inf
        index = np.where(distance == np.amin(distance))[0]
        clusters[index] = index[0]
        data[np.where(clusters == index[0])] = np.mean(data[index].T, axis=1)
    return normalize_groups(clusters)


def meu_ward(k, input_data):
    data = np.copy(input_data)
    classes = [[] for _ in range(k)]
    clusters = groups(data, k)
    for g, sample in zip(clusters, input_data):
        classes[g].append(sample)
    classes = [np.array(c) for c in classes]
    return np.array(classes), clusters


def get_min_dist_elements(data):
    distances = cdist(data, data)
    distances[distances == 0] = np.inf
    index = np.unravel_index(distances.argmin(), distances.shape)
    return index


def join_min_elements(data, index, groups):
    stacked = np.stack((data[index[0]], data[index[1]]))
    mean = stacked.mean(axis=1)
    g0 = groups[index[0]]
    g1 = groups[index[1]]
    for i, g in enumerate(groups):
        if g == g0 or g == g1:
            groups[i] = g0
            data[i] = mean


def get_num_groups(groups):
    return np.unique(groups).shape[0]


def normalize_groups(groups):
    for i, g in enumerate(np.unique(groups)):
        groups[groups == g] = i
    return groups


def ward_group_table(k, input_data):
    data = np.copy(input_data)
    groups = np.arange(data.shape[0])
    num_groups = data.shape[0]
    while num_groups > k:
        index = get_min_dist_elements(data)
        join_min_elements(data, index, groups)
        num_groups = get_num_groups(groups)

    return normalize_groups(groups)


def ward(k, data):
    clusters = [[] for _ in range(k)]
    groups = ward_group_table(k, data)
    for g, sample in zip(groups, data):
        clusters[g].append(sample)
    return clusters


def main():
    sd = 0.3
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
    ks = [2,4,8]
    for k in ks:
        colors = random_colors(k)
        # ward(4, data_con)
        ward = meu_ward(k, data_con)
        scatter_plot(
            data_con.T,
            [colors[u] for u in ward[1]],
            np.array([np.mean(c.T, axis=1) for c in ward[0]]).T
        )


if __name__ == '__main__':
    main()
