import numpy as np
from scipy.spatial.distance import cdist


class FuzzyKMeans:
    MAX_ITERATIONS = 1000

    def __init__(self, data, K, err):
        """
        data: any: Array any-size
        K: int: Number of clusters
        err: float: maximal error for difference between clusters centers
        """
        self._data = data
        self._k = K
        self._err = err

    @staticmethod
    def init_cluster_center(data, k):
        return np.array([data[np.random.randint(data.shape[0])] for _ in range(k)])

    @staticmethod
    def center_dist(data, c):
        return cdist(data, c) ** 2

    @staticmethod
    def membership_matrix(distances):
        return (distances == distances.min(axis=1)[:, None]).astype(int)

    @staticmethod
    def recalculate_centers(membership_m, data):
        # Recalcular centros
        number_points_per_class = np.sum(membership_m, axis=0)
        ## Calcular G
        G = np.expand_dims(number_points_per_class, 1) \
            .repeat((data.T @ membership_m).T.shape[-1], 1)
        return (data.T @ membership_m).T / G

    @staticmethod
    def normalize_membership_matrix(m):
        return np.array([
            np.argmax(p) for p in m
        ])

    def fkm(self):
        # Inicia K centros aleatoriamente
        c = self.init_cluster_center(self._data, self._k)
        membership_m = None
        i = None
        aux_J = None
        for i in range(self.MAX_ITERATIONS):
            # Calcula a Dist
            d = self.center_dist(self._data, c)
            # Aloca cada ponto a um centro
            membership_m = self.membership_matrix(d)

            # Recalcular os centros
            c = self.recalculate_centers(membership_m, self._data)

            # Calculo da tolerancia de parada
            J = np.diagonal(d.T @ membership_m).sum()
            if aux_J is not None and abs(aux_J - J) < self._err:
                return membership_m, c, i, J
            aux_J = J.copy()

        return membership_m, c, i
