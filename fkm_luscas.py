import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def init_cluster_center(data, n_clusters):
    centers = np.empty((n_clusters, data.shape[1]))
    for i in range(n_clusters):
        centers[i] = data[np.random.randint(data.shape[0])]
    return centers


def get_membership_matrix(center_distance_matrix):
    matrix = np.zeros(center_distance_matrix.shape)
    for sample_index, member_index in enumerate(center_distance_matrix.argmin(axis=1)):
        matrix[sample_index][member_index] = 1
    return matrix


def calculate_center(data, member_matrix):
    sample_per_cluster = np.sum(member_matrix, axis=0)[:, None]
    sum_matrix = member_matrix.T @ data
    # função tile expande o vetor de quantidaes para ter o mesmo tamanho da matriz de somas
    divider_matrix = np.tile(sample_per_cluster, data.shape[1])
    # Se por algum otivo algum cluster estiver vazio, ao invés de dividir por zero divide por 1
    divider_matrix[divider_matrix == 0] = 1
    return sum_matrix / divider_matrix


def assign_clusters(data, center_distance_matrix):
    clusters = list(map(lambda _: [], range(center_distance_matrix.shape[1])))
    for sample, distance in zip(data, center_distance_matrix):
        cluster_index = np.argmin(distance)
        if len(clusters[cluster_index]) == 0:
            clusters[cluster_index] = sample[None]
        else:
            clusters[cluster_index] = np.concatenate(
                (clusters[cluster_index], sample[None]), axis=0
            )
    return clusters


def print_k_means_result(plot, clusters, centers, cmap="Set1"):
    cmap = plt.cm.get_cmap(cmap, len(clusters))
    for i, cluster in enumerate(clusters):
        if len(cluster) and len(cluster[0]):
            plot.plot(cluster.T[0], cluster.T[1], "o", c=cmap(i))
        for i, center in enumerate(centers):
            plot.plot(center[0], center[1], "o", c=cmap(i), markersize=12, mec="black")


def k_means(n_clusters, data, max_iterations, plot=False):
    old_matrix = []
    centers = init_cluster_center(data, n_clusters)
    clusters = None
    membership_matrix = None
    for i in range(max_iterations):
        distances = cdist(data, centers)
        membership_matrix = get_membership_matrix(distances)
        if np.array_equal(old_matrix, membership_matrix):
            break
        old_matrix = membership_matrix
        centers = calculate_center(data, membership_matrix)
        clusters = assign_clusters(data, distances)
        if plot:
            if i % 3 == 0:
                plt.show()
                fig = plt.figure(figsize=plt.figaspect(0.35))
            plot = fig.add_subplot(1, 3, i % 3 + 1)
            print_k_means_result(plot, clusters, centers)
    if plot:
        plt.show()
    return membership_matrix, centers, clusters
