import numpy as np
from scipy.spatial.distance import cdist


def normalize_groups(groups):
    for i, g in enumerate(np.unique(groups)):
        groups[groups == g] = i
    return groups


def groups(data, k):
    clusters = np.arange(data.shape[0])
    while np.unique(clusters).shape[0] > k:
        distance = cdist(data, data)
        distance[distance == 0] = np.inf
        index = np.where(distance == np.amin(distance))[0]
        clusters[index] = index[0]
        data[np.where(clusters == index[0])] = np.mean(data[index].T, axis=1)
    return normalize_groups(clusters)


def ward(k, input_data):
    data = np.copy(input_data)
    classes = [[] for _ in range(k)]
    clusters = groups(data, k)
    for g, sample in zip(clusters, input_data):
        classes[g].append(sample)
    classes = [np.array(c) for c in classes]
    centers = np.array([np.mean(c, axis=0) for c in classes])
    return np.array(classes), clusters, centers
